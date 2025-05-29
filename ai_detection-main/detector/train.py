# -*- coding: utf-8 -*-
"""
Enhanced training script that adds
 • Perplexity
 • Burstiness (std‑dev of per‑chunk PPL)
 • Distinct‑n (lexical diversity, n = 1,2)
metrics to the original detector (roberta) pipeline.

If the optional --enable-mauve flag is passed and the mauve‑text
package is installed, a corpus‑level MAUVE score will also be
computed at the end of every validation epoch.  (This step can
consume several hundred MB of GPU RAM; it is disabled by default.)

Original file: train.py (2025‑05‑28)
"""

import argparse
import os
import math
from itertools import count
from collections import Counter

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    tokenization_utils_base as tokenization_utils,
)
import pandas as pd

# Optional — only used if --enable-mauve given
try:
    import mauve  # type: ignore
except ImportError:
    mauve = None

from .dataset import Corpus, EncodedDataset
from .utils import summary

###############################################################################
# Helper class for LM‑based statistics
###############################################################################
class LMScorer:
    """Compute perplexity, burstiness & distinct‑n on raw text."""

    def __init__(self, model_name: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.lm.eval()
        self.device = device

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        loss = self.lm(**enc, labels=enc["input_ids"]).loss
        return math.exp(loss.item())

    def burstiness(self, text: str, chunk: int = 30) -> float:
        toks = self.tokenizer.encode(text)
        if len(toks) < chunk * 2:
            return 0.0
        ppl_vals = [
            self.perplexity(self.tokenizer.decode(toks[i : i + chunk]))
            for i in range(0, len(toks), chunk)
        ]
        return torch.tensor(ppl_vals).std().item()

    @staticmethod
    def _distinct_n(tokens, n) -> float:
        if len(tokens) == 0:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return len(set(ngrams)) / max(len(ngrams), 1)

    def distinct_metrics(self, text: str):
        toks = self.tokenizer.tokenize(text)
        return {
            "distinct1": self._distinct_n(toks, 1),
            "distinct2": self._distinct_n(toks, 2),
        }

###############################################################################
# Dataset loader (unchanged apart from style tweaks)
###############################################################################

def load_datasets(
    data_dir,
    real_dataset,
    fake_datasets,  # list[str]
    tokenizer,
    batch_size,
    max_sequence_length,
    random_sequence_length,
    epoch_size=None,
    token_dropout=None,
    seed=None,
):
    real_corpus = Corpus(real_dataset, data_dir=data_dir)
    real_train, real_valid = real_corpus.train, real_corpus.valid

    # support multiple fake corpora
    if isinstance(fake_datasets, str):
        fake_datasets = [fake_datasets]
    fake_corpora = [Corpus(name, data_dir=data_dir) for name in fake_datasets]
    fake_train = sum([c.train for c in fake_corpora], [])
    fake_valid = sum([c.valid for c in fake_corpora], [])

    Sampler = RandomSampler
    min_seq_len = 10 if random_sequence_length else None

    train_dataset = EncodedDataset(
        real_train,
        fake_train,
        tokenizer,
        max_sequence_length,
        min_seq_len,
        epoch_size,
        token_dropout,
        seed,
    )
    train_loader = DataLoader(train_dataset, batch_size, sampler=Sampler(train_dataset))

    val_dataset = EncodedDataset(real_valid, fake_valid, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=Sampler(val_dataset))
    return train_loader, val_loader

###############################################################################
# Core training / validation loops (train unchanged)
###############################################################################

def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    return (classification == labels).float().sum().item()


def train(model: nn.Module, optimizer, device: str, loader: DataLoader, desc="Train"):
    model.train()
    train_acc = train_loss = train_ep_size = 0
    with tqdm(loader, desc=desc) as loop:
        for texts, masks, labels in loop:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            bsz = texts.size(0)
            optimizer.zero_grad()
            outputs = model(texts, attention_mask=masks, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            loss.backward()
            optimizer.step()
            train_acc += accuracy_sum(logits, labels)
            train_loss += loss.item() * bsz
            train_ep_size += bsz
            loop.set_postfix(loss=loss.item(), acc=train_acc / train_ep_size)
    return {
        "train/accuracy": train_acc,
        "train/epoch_size": train_ep_size,
        "train/loss": train_loss,
    }

###############################################################################
# Validation with new metrics
###############################################################################

def validate(
    model: nn.Module,
    device: str,
    loader: DataLoader,
    scorer: LMScorer,
    tokenizer: RobertaTokenizer,
    votes: int = 1,
    desc: str = "Validation",
    enable_mauve: bool = False,
):
    model.eval()
    val_acc = val_loss = val_ep_size = 0

    # storage for text to compute corpus‑level metrics later (distinct/mauve)
    human_texts, ai_texts = [], []

    # pre‑load records for vote ensembling
    records = [record for v in range(votes) for record in loader]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with torch.no_grad(), tqdm(records, desc=desc) as loop:
        for example in loop:
            losses, logit_votes = [], []
            for texts, masks, labels in example:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                outputs = model(texts, attention_mask=masks, labels=labels)
                losses.append(outputs.loss)
                logit_votes.append(outputs.logits)

            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)
            bsz = texts.size(0)
            val_acc += accuracy_sum(logits, labels)
            val_loss += loss.item() * bsz
            val_ep_size += bsz
            loop.set_postfix(loss=loss.item(), acc=val_acc / val_ep_size)

            # decode raw text for LM stats
            decoded = tokenizer.batch_decode(texts, skip_special_tokens=True)
            for txt, lbl in zip(decoded, labels):
                if lbl.item() == 0:
                    human_texts.append(txt)
                else:
                    ai_texts.append(txt)

    # —— LM‑based statistics ——
    perplexities = [scorer.perplexity(t) for t in human_texts + ai_texts]
    burstiness_vals = [scorer.burstiness(t) for t in human_texts + ai_texts]
    distinct1_vals = [scorer.distinct_metrics(t)["distinct1"] for t in human_texts + ai_texts]
    distinct2_vals = [scorer.distinct_metrics(t)["distinct2"] for t in human_texts + ai_texts]

    # aggregate
    metrics = {
        "validation/accuracy": val_acc,
        "validation/epoch_size": val_ep_size,
        "validation/loss": val_loss,
        "validation/perplexity": sum(perplexities) / len(perplexities),
        "validation/burstiness": sum(burstiness_vals) / len(burstiness_vals),
        "validation/distinct1": sum(distinct1_vals) / len(distinct1_vals),
        "validation/distinct2": sum(distinct2_vals) / len(distinct2_vals),
    }

    # —— optional MAUVE ——
    if enable_mauve and mauve is not None and human_texts and ai_texts:
        mauve_res = mauve.compute_mauve(
            p_text=ai_texts,
            q_text=human_texts,
            device_id=0 if device.startswith("cuda") else -1,
            max_text_length=256,
        )
        metrics["validation/mauve"] = mauve_res.mauve
    return metrics

###############################################################################
# Main runner (adds new CLI flags & CSV logging columns)
###############################################################################

def run(
    *,
    eval_lm="gpt2-xl",
    fake_datasets="xl-1542M-k40",
    enable_mauve=False,
    **kwargs,
):
    # —— Parse existing kwargs ——
    device = kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = kwargs.get("batch_size", 24)
    max_epochs = kwargs.get("max_epochs")
    max_seq_len = kwargs.get("max_sequence_length", 128)
    random_seq_len = kwargs.get("random_sequence_length", False)

    # —— Seed etc. ——
    torch.manual_seed(kwargs.get("seed") or 42)

    # —— Model & tokenizer ——
    model_name = "roberta-large" if kwargs.get("large") else "roberta-base"
    tokenization_utils.logger.setLevel("ERROR")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)

    # —— Optional checkpoint resume ——
    if (model_path := kwargs.get("model_path")) is not None:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

    summary(model)

    # —— Data ——
    train_loader, val_loader = load_datasets(
        data_dir=kwargs.get("data_dir", "data"),
        real_dataset=kwargs.get("real_dataset", "human_data/webtext"),
        fake_datasets=fake_datasets,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_sequence_length=max_seq_len,
        random_sequence_length=random_seq_len,
        epoch_size=kwargs.get("epoch_size"),
        token_dropout=kwargs.get("token_dropout"),
        seed=kwargs.get("seed"),
    )

    # —— Optim ——
    optimizer = Adam(model.parameters(), lr=kwargs.get("learning_rate", 2e-5), weight_decay=kwargs.get("weight_decay", 0))

    # —— LM scorer ——
    scorer = LMScorer(eval_lm, device)

    # —— Logging ——
    logdir = os.environ.get("OPENAI_LOGDIR", "logs")
    os.makedirs(logdir, exist_ok=True)
    writer = None
    if device.startswith("cuda"):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(logdir)

    excel_path = os.path.join(logdir, "training_logs.xlsx")
    log_rows = []

    best_val_acc = 0
    patience_counter = 0
    patience = kwargs.get("patience", 5)

    epoch_iter = count(1) if max_epochs is None else range(1, max_epochs + 1)
    for epoch in epoch_iter:
        train_metrics = train(model, optimizer, device, train_loader, f"Epoch {epoch}")
        val_metrics = validate(
            model,
            device,
            val_loader,
            scorer,
            tokenizer,
            votes=1,
            enable_mauve=enable_mauve,
        )

        # Normalise
        train_metrics["train/accuracy"] /= train_metrics["train/epoch_size"]
        train_metrics["train/loss"] /= train_metrics["train/epoch_size"]
        val_metrics["validation/accuracy"] /= val_metrics["validation/epoch_size"]
        val_metrics["validation/loss"] /= val_metrics["validation/epoch_size"]

        # —— Merge & log ——
        merged = {"epoch": epoch, **train_metrics, **val_metrics}
        log_rows.append(merged)
        pd.DataFrame(log_rows).to_excel(excel_path, index=False)
        print(f"Epoch {epoch} logged ✏️")

        if writer is not None:
            for k, v in merged.items():
                if k not in {"epoch", "train/epoch_size", "validation/epoch_size"}:
                    writer.add_scalar(k, v, epoch)

        # —— Early‑stopping & checkpoints ——
        cur_val_acc = val_metrics["validation/accuracy"]
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": kwargs,
                },
                os.path.join(logdir, "best-model.pt"),
            )
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered 🚦")
            break

    print(f"Training complete ✔️  Logs saved to {excel_path}")

###############################################################################
# Entry‑point CLI
###############################################################################

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # —— original args ——
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--device")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--max-sequence-length", type=int, default=128)
    p.add_argument("--random-sequence-length", action="store_true")
    p.add_argument("--epoch-size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--real-dataset", default="human_data/webtext")
    p.add_argument("--fake-datasets", nargs="+", default=["xl-1542M-k40"])
    p.add_argument("--token-dropout", type=float)
    p.add_argument("--large", action="store_true")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--model-path")

    # —— new args ——
    p.add_argument("--eval-lm", default="gpt2-xl", help="LM used for PPL/Burstiness computation")
    p.add_argument("--enable-mauve", action="store_true", help="Compute MAUVE (requires mauve‑text)")

    args = p.parse_args()
    run(**vars(args))