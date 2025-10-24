# detector/train_fixed.py — Speed-first preset (WSL/RTX3070 friendly)
# - RoBERTa-base, L=96, B=64 (default)
# - Tokenize with num_proc -> auto save/load pretokenized cache
# - SDPA(Flash) on, AMP, TF32, fused AdamW, non_blocking .to()
# - num_workers=0 (WSL2 안전)
# - data_tag='auto' → 모델/길이/데이터셋명 기반 캐시 자동 생성
# - ✅ tqdm에 train 누적 accuracy, loss, lr 표시

import argparse, os, random, time, re
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.backends.cuda import sdp_kernel
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, get_linear_schedule_with_warmup
)
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# ---- Environment / cuDNN / TF32 --------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---- Utils ------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sanitize(s: str) -> str:
    s = s.replace('/', '_').replace(':', '_')
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', s)

def leaf_name(path_like: str) -> str:
    return sanitize(path_like.strip('/').split('/')[-1])

def load_pair(prefix, split):
    return load_dataset('json', data_files=f"{prefix}.{split}.jsonl", split='train', cache_dir='./cache')

def cache_path(args) -> str:
    model_tag = sanitize(args.model)
    if args.data_tag is None or args.data_tag.lower() == 'auto':
        real_tag = leaf_name(args.real_dataset)
        fake_tag = leaf_name(args.fake_dataset)
        auto_tag = f"{real_tag}__{fake_tag}__v1"
    else:
        auto_tag = sanitize(args.data_tag)
    return os.path.join(args.tok_cache_root, f"{model_tag}__L{args.max_seq_len}__{auto_tag}")

# ---- Dataset build (auto-cache) ---------------------------------------------
def build_tokenized_datasets(args, tokenizer):
    keep = ['input_ids', 'attention_mask', 'labels']
    path = cache_path(args)
    if os.path.exists(path):
        print(f"[cache] loading pretokenized datasets from: {path}")
        dd = load_from_disk(path)
        dd = DatasetDict({k: v.with_format('torch') for k, v in dd.items()})
        return dd

    print("[cache] not found. Building tokenized datasets...")
    real_train = load_pair(os.path.join(args.data_dir, args.real_dataset), 'train').map(lambda x: {'labels': 0})
    fake_train = load_pair(os.path.join(args.data_dir, args.fake_dataset), 'train').map(lambda x: {'labels': 1})
    real_val   = load_pair(os.path.join(args.data_dir, args.real_dataset), 'valid').map(lambda x: {'labels': 0})
    fake_val   = load_pair(os.path.join(args.data_dir, args.fake_dataset), 'valid').map(lambda x: {'labels': 1})

    train = concatenate_datasets([real_train, fake_train])
    valid = concatenate_datasets([real_val, fake_val])

    if args.limit_train > 0: train = train.select(range(min(args.limit_train, len(train))))
    if args.limit_valid > 0: valid = valid.select(range(min(args.limit_valid, len(valid))))

    nproc = args.num_proc if args.num_proc > 0 else max(1, (os.cpu_count() or 1) // 2)
    print(f"[tokenize] num_proc={nproc}, max_len={args.max_seq_len}")

    def tok(ex): return tokenizer(ex['text'], truncation=True, max_length=args.max_seq_len)

    train = train.map(tok, batched=True, num_proc=nproc,
                      remove_columns=[c for c in train.column_names if c not in keep]).with_format('torch')
    valid = valid.map(tok, batched=True, num_proc=nproc,
                      remove_columns=[c for c in valid.column_names if c not in keep]).with_format('torch')

    dd = DatasetDict({'train': train, 'valid': valid})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dd.save_to_disk(path)
    print(f"[cache] saved pretokenized datasets to: {path}")
    return dd

def build_loaders(dd, tokenizer, args, device):
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if device == 'cuda' else None)
    train_ds, valid_ds = dd['train'], dd['valid']

    train_loader = DataLoader(
        train_ds, sampler=RandomSampler(train_ds),
        batch_size=args.batch_size, collate_fn=collator, pin_memory=True,
        num_workers=args.num_workers, persistent_workers=False
    )
    valid_loader = DataLoader(
        valid_ds, sampler=SequentialSampler(valid_ds),
        batch_size=args.eval_batch_size, collate_fn=collator, pin_memory=True,
        num_workers=args.num_workers, persistent_workers=False
    )
    print(f"[data] train={len(train_ds):,}  valid={len(valid_ds):,}  "
          f"steps/epoch={len(train_loader):,} (bs={args.batch_size})")
    return train_loader, valid_loader

# ---- Evaluation --------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); total_loss=0.0; n=0
    all_preds, all_labels = [], []
    for batch in tqdm(loader, total=len(loader), desc="Valid", ncols=120, leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch); loss, logits = out.loss, out.logits
        bsz = batch['labels'].size(0); total_loss += loss.item() * bsz; n += bsz
        all_preds.extend(logits.argmax(-1).detach().cpu().tolist())
        all_labels.extend(batch['labels'].detach().cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return {'loss': total_loss / max(n, 1), 'acc': acc, 'precision': p, 'recall': r, 'f1': f1}

# ---- Train -------------------------------------------------------------------
def train(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # SDPA (Flash/MemEfficient) 우선 사용 — bert/roberta에서 특히 유효
    try:
        if args.flash_sdp:
            sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
            print("SDPA: flash/mem_efficient enabled")
    except Exception as e:
        print("SDPA toggle failed:", e)

    # 모델/토크나이저
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    # 데이터셋(캐시 자동) → 로더
    dd = build_tokenized_datasets(args, tokenizer)
    train_loader, valid_loader = build_loaders(dd, tokenizer, args, device)

    # 옵티마이저 (fused AdamW 지원 시 자동 사용)
    use_fused = torch.cuda.is_available()
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                      eps=1e-8, fused=use_fused)
        if use_fused: print("AdamW fused=True")
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

    steps_total = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * steps_total), steps_total)
    scaler = GradScaler(enabled=(device == 'cuda'))

    best_f1 = -1.0; patience = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train(); running_loss = 0.0; t0 = time.time()
        running_correct = 0; running_seen = 0  # ✅ 누적 정확도 집계
        bar = tqdm(train_loader, total=len(train_loader), desc=f"Train e{epoch}", ncols=120)

        for step, batch in enumerate(bar, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast(device_type=device, enabled=(device == 'cuda')):
                out = model(**batch)
                loss = out.loss / args.grad_accum
                # ✅ 배치 예측으로 누적 정확도 업데이트(가벼움)
                preds = out.logits.detach().argmax(dim=-1)
                bsz = batch['labels'].size(0)
                running_correct += (preds == batch['labels']).sum().item()
                running_seen += bsz

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True); scheduler.step()

            running_loss += loss.item() * args.grad_accum

            # ✅ 지정 간격으로만 tqdm 갱신(오버헤드 절약)
            if step % args.log_every == 0 or step == 1:
                train_acc = running_correct / max(running_seen, 1)
                bar.set_postfix(loss=f"{running_loss/step:.4f}",
                                acc=f"{train_acc:.4f}",
                                lr=f"{scheduler.get_last_lr()[0]:.2e}")

        val = evaluate(model, valid_loader, device)
        mins = (time.time() - t0) / 60.0
        print(f"[Epoch {epoch}] train_loss={running_loss/len(train_loader):.4f} | "
              f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} val_f1={val['f1']:.4f} | "
              f"time={mins:.1f}m")

        if val['f1'] > best_f1 + 1e-4:
            best_f1 = val['f1']; patience = args.patience
            os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
            torch.save({'state_dict': model.state_dict(), 'model_name': args.model}, args.ckpt)
            print(f"★ Saved best model to {args.ckpt} (f1={best_f1:.4f}, model={args.model})")
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping."); break

# ---- Defaults (최소 인자로 돌도록 구성) ---------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    # 데이터 루트 & 파일 prefix
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--real-dataset', default='human_data/webtext')
    ap.add_argument('--fake-dataset', default='ai_data/gemini/gemini')  # 기본: gemini

    # 내 추천 최단 루트 (속도 우선)
    ap.add_argument('--model', default='roberta-base')          # 필요시 microsoft/deberta-v3-base 로 변경, roberta-base
    ap.add_argument('--max-seq-len', type=int, default=96)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--eval-batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--warmup-ratio', type=float, default=0.1)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--max-grad-norm', type=float, default=1.0)
    ap.add_argument('--ckpt', default='./logs/best-model.pt')
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)

    # WSL 안전 구동
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--log-every', type=int, default=50)

    # 토크나이즈 캐시(자동 사용) 관련
    ap.add_argument('--num-proc', type=int, default=6)
    ap.add_argument('--tok-cache-root', default='./cache/tok')
    ap.add_argument('--data-tag', default='auto')               # ★ 자동 태깅
    ap.add_argument('--limit-train', type=int, default=0)
    ap.add_argument('--limit-valid', type=int, default=0)

    # SDPA(Flash/MemEfficient) 커널 사용
    ap.add_argument('--flash-sdp', action='store_true', default=True)
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)