# =============================================================================
# detector/train_fixed.py
# -----------------------------------------------------------------------------
# 📌 목적(purpose)
#   - "인간 vs AI(생성형 모델) 텍스트" 이진 분류기를 빠르고 안정적으로 학습하기 위한
#     스크립트. WSL2 + RTX 3070(8GB) 환경에서 속도/메모리/안정성을 최우선으로 튜닝.
#
# 📦 모델(Model)
#   - 기본: RoBERTa-base (Transformer Encoder 기반, bidirectional attention)
#     · 사전학습(objective): Masked Language Modeling
#     · 미세조정(task): 이진 분류(head: hidden_state → [CLS] → Linear(num_labels=2))
#   - 선택: microsoft/deberta-v3-base (Disentangled self-attention; 더 무거울 수 있음)
#
# 🧵 파이프라인(Pipeline)
#   1) 데이터 로딩: human(real) / ai(fake) jsonl (train/valid) → 합치고 레이블 부여(0/1)
#   2) 토크나이즈: HuggingFace Datasets의 .map(num_proc=6) 병렬 처리
#      · 결과를 디스크 캐시(save_to_disk) → 동일 설정이면 다음 실행에서 즉시 로드
#   3) DataLoader: WSL2 안정성을 위해 num_workers=0, pad_to_multiple_of=8 (AMP 유리)
#   4) 학습: AMP(FP16) + TF32, Fused AdamW, SDPA(Flash/MemEfficient) 선호
#      · tqdm에 loss/acc/lr 실시간 표시
#      · warmup + linear decay 스케줄러
#      · gradient clipping + early stopping
#   5) 평가: valid에서 loss/accuracy/macro-F1 산출 → 최고 F1 모델 저장
#
# ⚙️ 성능(throughput) 팁
#   - L(시퀀스 길이)↓ → 계산량 ~L^2로 줄어듦. 기본 L=96 (추론에선 슬라이딩 윈도우로 보완)
#   - AMP/TF32/SDPA/AdamW(fused) → 1 epoch ~25~30분(500k 샘플, bs=64) 수준 가능
#   - 캐시 자동: 모델/길이/데이터 조합별로 폴더가 달라져 섞이지 않음
#
# 🧪 사용법
#   $ python detector/train_fixed.py
#   (필요시) --model, --max-seq-len, --batch-size, --epochs 등 CLI 인자로 덮어쓰기 가능
# =============================================================================

import argparse, os, random, time, re, contextlib               # 기본 유틸 · 컨텍스트
import numpy as np                                              # 난수 고정/간단 계산
import torch                                                    # PyTorch 핵심
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler  # 로더/샘플러
from torch.nn.attention import sdpa_kernel                      # ✅ 신규 SDPA 컨텍스트
from transformers import (                                      # HF Transformers
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, get_linear_schedule_with_warmup
)
from datasets import (                                          # HF Datasets
    load_dataset, concatenate_datasets, DatasetDict, load_from_disk
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support # 지표
from torch.amp import GradScaler, autocast                      # AMP(자동 혼합 정밀)
from tqdm import tqdm                                           # 진행률 출력
from transformers.utils import logging as hf_logging            # HF 로그 레벨 제어
hf_logging.set_verbosity_error()                                # HF 경고 과다 출력 방지

# -------------------- 런타임 튜닝: 토크나이저/TF32/cuDNN --------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"                  # 토크나이저 thread 폭주 방지
torch.backends.cuda.matmul.allow_tf32 = True                    # TF32(암페어↑) 허용 → 속도↑
torch.backends.cudnn.allow_tf32 = True                          # cuDNN에서도 TF32 허용
torch.backends.cudnn.benchmark = True                           # 입력 크기 고정 시 커널 auto-tune
try:
    torch.set_float32_matmul_precision("high")                  # PyTorch 2.x 권장 설정
except Exception:
    pass

# ------------------------------ 유틸 함수들 ---------------------------------
def set_seed(seed: int = 42):
    """실험 재현성을 위한 시드 고정."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sanitize(s: str) -> str:
    """경로/파일명 안전 문자열로 정제."""
    s = s.replace('/', '_').replace(':', '_')
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', s)

def leaf_name(path_like: str) -> str:
    """'a/b/c' → 'c' (마지막 토큰만)"""
    return sanitize(path_like.strip('/').split('/')[-1])

def load_pair(prefix: str, split: str):
    """
    JSONL 로드: {prefix}.{split}.jsonl
    · HF datasets가 내부 캐시를 활용해 빠르게 로드
    """
    return load_dataset('json', data_files=f"{prefix}.{split}.jsonl",
                        split='train', cache_dir='./cache')

def cache_path(args) -> str:
    """
    토크나이즈 결과를 저장/로드할 디스크 경로 생성.
    data_tag='auto'면 (real_leaf__fake_leaf__v1) 패턴으로 자동 태깅.
    """
    model_tag = sanitize(args.model)
    if args.data_tag is None or args.data_tag.lower() == 'auto':
        real_tag = leaf_name(args.real_dataset)                 # 예: 'webtext'
        fake_tag = leaf_name(args.fake_dataset)                 # 예: 'gemini'
        auto_tag = f"{real_tag}__{fake_tag}__v1"               # 필요시 --data-tag 로 v2, v3…
    else:
        auto_tag = sanitize(args.data_tag)
    # 예: ./cache/tok/roberta-base__L96__webtext__gemini__v1
    return os.path.join(args.tok_cache_root,
                        f"{model_tag}__L{args.max_seq_len}__{auto_tag}")

# -------------------------- 데이터셋 빌드 & 캐시 ----------------------------
def build_tokenized_datasets(args, tokenizer):
    """
    1) 캐시가 있으면 load_from_disk
    2) 없으면 raw jsonl 로드 → 라벨링 → 병합 → (선택적)서브샘플 → 토크나이즈(map)
       → save_to_disk → 반환
    """
    keep = ['input_ids', 'attention_mask', 'labels']            # 훈련에 필요한 열만 유지
    path = cache_path(args)                                     # 캐시 경로 계산

    if os.path.exists(path):                                    # 1) 캐시 히트
        print(f"[cache] loading pretokenized datasets from: {path}")
        dd = load_from_disk(path)                               # 디스크에서 로드
        dd = DatasetDict({k: v.with_format('torch') for k, v in dd.items()})  # 텐서 포맷
        return dd

    # 2) 캐시 없음 → 원본 로드 후 전처리
    print("[cache] not found. Building tokenized datasets...")

    # split별 jsonl 읽고 라벨 부여: human=0, ai=1
    real_train = load_pair(os.path.join(args.data_dir, args.real_dataset), 'train').map(lambda x: {'labels': 0})
    fake_train = load_pair(os.path.join(args.data_dir, args.fake_dataset), 'train').map(lambda x: {'labels': 1})
    real_val   = load_pair(os.path.join(args.data_dir, args.real_dataset), 'valid').map(lambda x: {'labels': 0})
    fake_val   = load_pair(os.path.join(args.data_dir, args.fake_dataset), 'valid').map(lambda x: {'labels': 1})

    # 사람/AI 합치기(순서는 sampler가 섞어줌)
    train = concatenate_datasets([real_train, fake_train])
    valid = concatenate_datasets([real_val,   fake_val])

    # 빠른 실험을 위한 샘플 수 제한(0이면 전체)
    if args.limit_train > 0:
        train = train.select(range(min(args.limit_train, len(train))))
    if args.limit_valid > 0:
        valid = valid.select(range(min(args.limit_valid, len(valid))))

    # 멀티프로세싱 토크나이즈(WSL2에서도 datasets.map의 num_proc은 잘 작동)
    nproc = args.num_proc if args.num_proc > 0 else max(1, (os.cpu_count() or 1) // 2)
    print(f"[tokenize] num_proc={nproc}, max_len={args.max_seq_len}")

    # 토크나이저 콜백: truncation=True + max_length=지정
    def tok(ex): return tokenizer(ex['text'], truncation=True, max_length=args.max_seq_len)

    # 실제 토크나이즈: 열 정리 후 torch 텐서 포맷으로 바꿈
    train = train.map(tok, batched=True, num_proc=nproc,
                      remove_columns=[c for c in train.column_names if c not in keep]
                     ).with_format('torch')
    valid = valid.map(tok, batched=True, num_proc=nproc,
                      remove_columns=[c for c in valid.column_names if c not in keep]
                     ).with_format('torch')

    # 디스크에 저장(다음 실행에서 즉시 로드)
    dd = DatasetDict({'train': train, 'valid': valid})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dd.save_to_disk(path)
    print(f"[cache] saved pretokenized datasets to: {path}")
    return dd

def build_loaders(dd, tokenizer, args, device):
    """
    토크나이즈된 DatasetDict → DataLoader 생성.
    pad_to_multiple_of=8: AMP(FP16)에서 텐서 얼라인먼트로 커널 효율 상승.
    """
    collator = DataCollatorWithPadding(tokenizer,
                                       pad_to_multiple_of=8 if device == 'cuda' else None)
    train_ds, valid_ds = dd['train'], dd['valid']

    # 학습은 RandomSampler(셔플), 검증은 SequentialSampler(순차)
    train_loader = DataLoader(train_ds, sampler=RandomSampler(train_ds),
                              batch_size=args.batch_size, collate_fn=collator,
                              pin_memory=True, num_workers=args.num_workers,
                              persistent_workers=False)          # WSL2: workers=0 안전

    valid_loader = DataLoader(valid_ds, sampler=SequentialSampler(valid_ds),
                              batch_size=args.eval_batch_size, collate_fn=collator,
                              pin_memory=True, num_workers=args.num_workers,
                              persistent_workers=False)

    print(f"[data] train={len(train_ds):,}  valid={len(valid_ds):,}  "
          f"steps/epoch={len(train_loader):,} (bs={args.batch_size})")
    return train_loader, valid_loader

# ------------------------------ 평가 루틴 -----------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    """
    검증 루프: loss/acc/macro-F1 산출.
    @torch.no_grad(): 평가 중 그래프·메모리 사용 최소화(속도↑).
    """
    model.eval()                                                # 드롭아웃/정규화 고정
    total_loss = 0.0; n = 0                                     # 누적 손실/샘플 수
    all_preds, all_labels = [], []                              # 예측/정답 버퍼

    for batch in tqdm(loader, total=len(loader), desc="Valid", ncols=120, leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}  # H→D 복사 겹치기
        out = model(**batch)                                    # forward
        loss, logits = out.loss, out.logits                     # 손실·로짓
        bsz = batch['labels'].size(0)
        total_loss += loss.item() * bsz; n += bsz               # 평균 손실 계산용
        all_preds.extend(logits.argmax(-1).detach().cpu().tolist())   # 예측 라벨
        all_labels.extend(batch['labels'].detach().cpu().tolist())    # 정답 라벨

    acc = accuracy_score(all_labels, all_preds)                 # 정확도
    p, r, f1, _ = precision_recall_fscore_support(              # 매크로 P/R/F1
        all_labels, all_preds, average='macro', zero_division=0
    )
    return {'loss': total_loss / max(n, 1), 'acc': acc, 'precision': p, 'recall': r, 'f1': f1}

# ------------------------------ 학습 루틴 -----------------------------------
def train(args):
    set_seed(args.seed)                                         # 재현성
    device = 'cuda' if torch.cuda.is_available() else 'cpu'     # 장치 선택
    print("Device:", device)

    # SDPA(Flash/MemEfficient) 컨텍스트 구성
    # - 신규 API(torch.nn.attention.sdpa_kernel)는 컨텍스트 매니저로,
    #   해당 with 블록에서만 커널 우선순위가 적용됨.
    sdpa_ctx = sdpa_kernel(enable_flash=bool(args.flash_sdp),
                           enable_mem_efficient=bool(args.flash_sdp),
                           enable_math=not bool(args.flash_sdp)) if device == 'cuda' else contextlib.nullcontext()

    # 토크나이저/모델 로드 (num_labels=2 → 이진 분류 헤드가 붙은 상태)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    # 데이터셋(캐시 자동) 생성 → DataLoader
    dd = build_tokenized_datasets(args, tokenizer)
    train_loader, valid_loader = build_loaders(dd, tokenizer, args, device)

    # 옵티마이저: Fused AdamW(가능 시) → GPU step 오버헤드 감소
    use_fused = torch.cuda.is_available()
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay, eps=1e-8,
                                      fused=use_fused)
        if use_fused:
            print("AdamW fused=True")
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay, eps=1e-8)

    # 스케줄러: linear decay + warmup
    steps_total = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                int(args.warmup_ratio * steps_total),
                                                steps_total)
    scaler = GradScaler(enabled=(device == 'cuda'))            # AMP 스케일러

    best_f1 = -1.0; patience = args.patience                   # Early stopping 상태
    with sdpa_ctx:                                             # ✅ 학습·평가 전체 구간에 SDPA 적용
        if args.flash_sdp and device == 'cuda':
            print("SDPA: flash/mem_efficient enabled")

        for epoch in range(1, args.epochs + 1):                # 에폭 루프
            model.train()                                      # 학습 모드
            running_loss = 0.0; t0 = time.time()               # 로깅용 누적값
            running_correct = 0; running_seen = 0              # 진행 중 정확도 집계
            bar = tqdm(train_loader, total=len(train_loader),
                      desc=f"Train e{epoch}", ncols=120)       # 진행률 바

            for step, batch in enumerate(bar, 1):              # 미니배치 루프
                # 비동기 장치 이동(non_blocking)으로 H→D 복사와 연산 겹치기
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                # AMP: Tensor Core/FP16 경로 사용 (속도↑, 메모리↓)
                with autocast(device_type=device, enabled=(device == 'cuda')):
                    out = model(**batch)                       # forward
                    loss = out.loss / args.grad_accum          # grad accumulation 대비 스케일

                    # 진행 중 정확도 계산(가벼움, 그래프 분리)
                    preds = out.logits.detach().argmax(dim=-1)
                    bsz = batch['labels'].size(0)
                    running_correct += (preds == batch['labels']).sum().item()
                    running_seen += bsz

                # 스케일된 그라디언트 역전파
                scaler.scale(loss).backward()

                # grad_accum 스텝마다 옵티마이저 업데이트
                if step % args.grad_accum == 0:
                    scaler.unscale_(optimizer)                 # clip 전에 unscale
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)
                    scaler.step(optimizer); scaler.update()    # 옵티마이저 스텝 + 스케일 업데이트
                    optimizer.zero_grad(set_to_none=True)      # grad 메모리 해제
                    scheduler.step()                           # 러닝레이트 스케줄

                running_loss += loss.item() * args.grad_accum  # 원래 스케일로 누적

                # 로그 갱신(너무 자주 갱신하면 오버헤드 → log_every 간격으로)
                if step % args.log_every == 0 or step == 1:
                    train_acc = running_correct / max(running_seen, 1)
                    bar.set_postfix(loss=f"{running_loss/step:.4f}",
                                    acc=f"{train_acc:.4f}",
                                    lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # 에폭 종료 → 검증
            val = evaluate(model, valid_loader, device)
            mins = (time.time() - t0) / 60.0
            print(f"[Epoch {epoch}] train_loss={running_loss/len(train_loader):.4f} | "
                  f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} val_f1={val['f1']:.4f} | "
                  f"time={mins:.1f}m")

            # 베스트 모델 저장(기준: macro-F1)
            if val['f1'] > best_f1 + 1e-4:
                best_f1 = val['f1']; patience = args.patience
                os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
                torch.save({'state_dict': model.state_dict(), 'model_name': args.model}, args.ckpt)
                print(f"★ Saved best model to {args.ckpt} (f1={best_f1:.4f}, model={args.model})")
            else:
                patience -= 1
                if patience <= 0:
                    print("Early stopping."); break            # 조기 종료

# --------------------------- 기본 인자(Default) -----------------------------
def build_argparser():
    ap = argparse.ArgumentParser()

    # 데이터 루트 & 파일 prefix (jsonl 경로 접두사)
    ap.add_argument('--data-dir', default='data')               # 루트 폴더
    ap.add_argument('--real-dataset', default='human_data/webtext')     # 사람 데이터 prefix
    ap.add_argument('--fake-dataset', default='ai_data/gemini/gemini')  # AI 데이터 prefix

    # 최적화된 기본값(속도·안정성 우선)
    ap.add_argument('--model', default='roberta-base')          # 필요시 deberta-v3-base로 비교
    ap.add_argument('--max-seq-len', type=int, default=96)      # L↓ → 속도↑(추론은 윈도우로 보완)
    ap.add_argument('--batch-size', type=int, default=64)       # 3070 8GB 권장값
    ap.add_argument('--eval-batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)           # early stopping이 있으니 과함 OK
    ap.add_argument('--lr', type=float, default=2e-5)           # AdamW 기본 학습률
    ap.add_argument('--weight-decay', type=float, default=0.01) # L2 정규화
    ap.add_argument('--warmup-ratio', type=float, default=0.1)  # 10% 워밍업
    ap.add_argument('--grad-accum', type=int, default=1)        # 메모리 여유시 1 유지
    ap.add_argument('--max-grad-norm', type=float, default=1.0) # 그라디언트 클립
    ap.add_argument('--ckpt', default='./logs/best-model.pt')   # 체크포인트 경로
    ap.add_argument('--patience', type=int, default=5)          # F1 개선 없을 때 허용 에폭 수
    ap.add_argument('--seed', type=int, default=42)             # 재현성

    # WSL2 안전 구동
    ap.add_argument('--num-workers', type=int, default=0)       # DataLoader 프로세스(WSL은 0 권장)
    ap.add_argument('--log-every', type=int, default=50)        # tqdm 업데이트 간격

    # 토크나이즈 캐시(자동 사용) 설정
    ap.add_argument('--num-proc', type=int, default=6)          # datasets.map 병렬 토크나이즈
    ap.add_argument('--tok-cache-root', default='./cache/tok')  # 캐시 루트 폴더
    ap.add_argument('--data-tag', default='auto')               # auto → 조합별 폴더 자동 생성
    ap.add_argument('--limit-train', type=int, default=0)       # 빠른 실험용 서브셋(0=전체)
    ap.add_argument('--limit-valid', type=int, default=0)

    # SDPA(Flash/MemEfficient) 사용(가능 시)
    ap.add_argument('--flash-sdp', action='store_true', default=True)
    return ap

# --------------------------------- 진입점 -----------------------------------
if __name__ == "__main__":
    args = build_argparser().parse_args()                       # 인자 파싱(기본값 위주)
    train(args)                                                 # 학습 시작