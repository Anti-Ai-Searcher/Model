#!/usr/bin/env python3
"""hf2jsonl.py  –  Convert HF LLM‑output datasets to JSONL
────────────────────────────────────────────────────────────
◎ 기능
  • Hugging Face 데이터셋을 *streaming* 으로 읽어
    train/valid/test(92/4/4%) 로 분할해 data/ai_data/ 폴더에 저장.
  • 현재 등록된 6개 코퍼스 모두 **퍼블릭** repo 만 사용.
  • ShareGPT52K 특수 파서는 제거 → 404/권한 문제 원천 차단.

Run examples
~~~~~~~~~~~~
$ python tools/hf2jsonl.py             # 6개 모두 변환
$ python tools/hf2jsonl.py --key gpt4  # 특정 key 하나만 변환
"""
from __future__ import annotations
import argparse, json, random, pathlib, itertools
from typing import Iterator, Dict, Any

from datasets import load_dataset

DATA_ROOT = pathlib.Path("data/ai_data")

# -----------------------------------------------------------
# 원하는 LLM‑출력 데이터셋 정의 (repo_id, split, text_col, group)
DATASETS = {
    "gpt35":  ("teknium/GPTeacher-ShareGPT_V3",     "train", "conversations",      "gpt35"),
    "gpt4":   ("openbmb/UltraInteract_sft",         "train", "response",           "gpt4"),
    "gpt_o1": ("Open-Orca/SlimOrca",                "train", "assistant_response", "gpt_o1"),
    "gpt_o3": ("vicgalle/creative-rubrics-gpt-4.5-o3-R1", "train", "output",      "gpt_o3"),
    "gemini15":("mlfoundations-dev/oh-dcft-v3.1-gemini-1.5-flash", "train", "content", "gemini15"),
    "gemini25":("bingxingliu/Gemini-2_5-Flash",     "train", "text",               "gemini25"),
}

# -----------------------------------------------------------
# Generic streaming iterator

def iter_records(repo: str, split: str, text_col: str) -> Iterator[Dict[str, Any]]:
    ds = load_dataset(repo, split=split, streaming=True)
    for i, rec in enumerate(ds):
        if text_col in rec and isinstance(rec[text_col], str):
            yield {"id": i, "text": rec[text_col].strip()}
        elif "conversations" in rec:  # 일부 Vicuna 형식 등
            txt = "\n".join(m.get("value", "") for m in rec["conversations"])
            yield {"id": i, "text": txt.strip()}
        else:  # fallback: 전체 json 직렬화
            yield {"id": i, "text": json.dumps(rec, ensure_ascii=False)}

# -----------------------------------------------------------
# Save helpers

def save_jsonl(records: list[dict[str, Any]], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✨ {len(records):,} → {path}")

# -----------------------------------------------------------

def make_splits(key: str, repo: str, split: str, text_col: str, group: str, seed: int = 42, pct: float = 0.04):
    print(f"\n=== [{key}] {repo}")
    recs = list(iter_records(repo, split, text_col))
    random.Random(seed).shuffle(recs)
    n = len(recs); n_val = int(n * pct)
    val   = recs[:n_val]
    test  = recs[n_val: n_val*2]
    train = recs[n_val*2:]

    base = DATA_ROOT / group
    save_jsonl(train, base / f"{key}.train.jsonl")
    save_jsonl(val,   base / f"{key}.valid.jsonl")
    save_jsonl(test,  base / f"{key}.test.jsonl")

# -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", help="convert only this key", default=None)
    args = ap.parse_args()

    subset = {args.key: DATASETS[args.key]} if args.key else DATASETS
    for k, (repo, split, text_col, group) in subset.items():
        make_splits(k, repo, split, text_col, group)

if __name__ == "__main__":
    main()