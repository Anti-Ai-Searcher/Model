# test_fixed.py
import argparse, os
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss, logits = out.loss, out.logits
        probs = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
        preds = logits.argmax(-1).detach().cpu().numpy()
        labels = batch['labels'].detach().cpu().numpy()
        m = labels.shape[0]
        total_loss += loss.item() * m; n += m
        all_probs.extend(probs); all_preds.extend(preds); all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return {'loss': total_loss/max(n,1), 'acc':acc, 'p':p, 'r':r, 'f1':f1,
            'probs':np.array(all_probs), 'preds':np.array(all_preds), 'labels':np.array(all_labels)}

def load_pair(prefix, split):
    return load_dataset('json', data_files=f"{prefix}.{split}.jsonl", split='train', cache_dir='./cache')

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.model_path, map_location=device)
    model_name = ckpt.get('model_name', 'roberta-base') # microsoft/deberta-v3-base OR roberta-base
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if device=='cuda' else None)

    real = load_pair(os.path.join(args.data_dir, args.real_dataset), 'test').map(lambda x:{'labels':0})
    fake = load_pair(os.path.join(args.data_dir, args.fake_dataset), 'test').map(lambda x:{'labels':1})
    ds = concatenate_datasets([real, fake])

    ds = ds.map(lambda ex: tokenizer(ex['text'], truncation=True, max_length=args.max_seq_len),
                batched=True, remove_columns=[c for c in ds.column_names if c not in ['input_ids','attention_mask','labels']])
    ds = ds.with_format('torch')
    loader = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=args.batch_size, collate_fn=collator)

    res = evaluate(model, loader, device)
    print(f"TEST | loss={res['loss']:.4f} acc={res['acc']:.4f} p={res['p']:.4f} r={res['r']:.4f} f1={res['f1']:.4f}")

    cm = confusion_matrix(res['labels'], res['preds'])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Human','Gemini-2.5'], yticklabels=['Human','Gemini-2.5']) # 여기 라벨 수정할 것
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=160)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', default='./logs/best-model.pt')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--real-dataset', default='human_data/webtext')
    ap.add_argument('--fake-dataset', default='ai_data/gemini/gemini')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--max-seq-len', type=int, default=256)
    args = ap.parse_args()
    main(args)
