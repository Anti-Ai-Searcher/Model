#!/usr/bin/env python3
# tools/csv2jsonl.py
"""
Robust CSV → JSONL converter
 • picks any column as text (default: 'text')
 • optional id column (auto-increment if missing)
 • keeps only {"id":…, "text":…} keys exactly like your GPT-2 jsonl
"""

import argparse, pathlib, json, pandas as pd

def main(csv_path, text_col="text", id_col=None, on_bad="warn"):
    csv_f = pathlib.Path(csv_path)
    df = pd.read_csv(csv_f, on_bad_lines=on_bad)           # pandas ≥1.3
    if text_col not in df.columns:
        raise KeyError(f"'{text_col}' column not found. "
                       f"Existing columns: {list(df.columns)}")

    # ── id 처리 ───────────────────────────────────────────
    if id_col and id_col in df.columns:
        ids = df[id_col]
    else:                         # 없으면 0,1,2…
        ids = range(len(df))

    out_f = csv_f.with_suffix(".jsonl")
    with out_f.open("w", encoding="utf-8") as f:
        for _id, txt in zip(ids, df[text_col].astype(str)):
            rec = {"id": int(_id), "text": txt.strip()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅  {len(df):,} rows  →  {out_f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_path",          help="source CSV file")
    p.add_argument("--text_col", default="text",
                   help="column that contains the raw text")
    p.add_argument("--id_col",   default=None,
                   help="column to use as id (optional)")
    p.add_argument("--on_bad",   default="warn",
                   choices=["warn", "skip", "error"],
                   help="how to handle malformed CSV lines (>=pandas 1.3)")
    main(**vars(p.parse_args()))
