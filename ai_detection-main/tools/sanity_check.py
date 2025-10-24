import torch
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm
import argparse
import time
from torch.amp import GradScaler, autocast

def run_sanity_check(args):
    """
    데이터 로딩 없이 GPU의 순수 연산 속도만 테스트합니다.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available. Cannot perform sanity check.")
        return
        
    print(f"Running sanity check on device: {device}")
    print(f"Batch Size: {args.batch_size}, Sequence Length: {args.seq_len}")

    # 1. 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = GradScaler()
    model.train()

    # 2. GPU 메모리 위에서 직접 가상 데이터 생성
    # 이 데이터는 디스크나 RAM을 거치지 않습니다.
    dummy_input_ids = torch.randint(0, 30000, (args.batch_size, args.seq_len), device=device)
    dummy_attention_mask = torch.ones(args.batch_size, args.seq_len, device=device)
    dummy_labels = torch.randint(0, 2, (args.batch_size,), device=device)

    # 3. 순수 연산 속도 측정
    start_time = time.time()
    loop = tqdm(range(args.num_steps), desc="Sanity Check Loop")
    
    for _ in loop:
        optimizer.zero_grad()

        with autocast(device_type=device):
            outputs = model(input_ids=dummy_input_ids,
                              attention_mask=dummy_attention_mask,
                              labels=dummy_labels)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_it = total_time / args.num_steps
    
    print("\n--- Sanity Check Result ---")
    print(f"Total steps: {args.num_steps}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per iteration (s/it): {avg_time_per_it:.4f} seconds")
    print("---------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='microsoft/deberta-v3-base')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--num-steps', type=int, default=100, help="Number of test iterations")
    args = parser.parse_args()
    run_sanity_check(args)