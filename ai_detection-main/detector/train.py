"""Training code for the detector model"""

import argparse
import os
import subprocess
import sys
from itertools import count
from multiprocessing import Process

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm
from transformers import *
import pandas as pd


from .dataset import Corpus,EncodedDataset
from .utils import summary


def load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                  max_sequence_length, random_sequence_length, epoch_size=None, token_dropout=None, seed=None):

    real_corpus = Corpus(real_dataset, data_dir=data_dir)

    if fake_dataset == "TWO":
        real_train, real_valid = real_corpus.train * 2, real_corpus.valid * 2
        fake_corpora = [Corpus(name, data_dir=data_dir) for name in ['xl-1542M', 'xl-1542M-k40']]
        fake_train = sum([corpus.train for corpus in fake_corpora], [])
        fake_valid = sum([corpus.valid for corpus in fake_corpora], [])
    else:
        fake_corpus = Corpus(fake_dataset, data_dir=data_dir)

        real_train, real_valid = real_corpus.train, real_corpus.valid
        fake_train, fake_valid = fake_corpus.train, fake_corpus.valid

    Sampler = RandomSampler

    min_sequence_length = 10 if random_sequence_length else None
    train_dataset = EncodedDataset(real_train, fake_train, tokenizer, max_sequence_length, min_sequence_length,
                                   epoch_size, token_dropout, seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=Sampler(train_dataset), num_workers=0)

    validation_dataset = EncodedDataset(real_valid, fake_valid, tokenizer)
    # subset_len = len(validation_dataset)/100
    # subset_validation_dataset = Subset(validation_dataset, range(int(subset_len)))
    # validation_loader = DataLoader(subset_validation_dataset, batch_size=1, sampler=Sampler(subset_validation_dataset))
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=Sampler(validation_dataset))

    return train_loader, validation_loader


def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()

        #  모델 = roBERTa    Adam       cpu or gpu   학습 데이터 로드     진행 표시줄
def train(model: nn.Module, optimizer, device: str, loader: DataLoader, desc='Train'):
    model.train() #모델을 train모드로

    #학습변수 초기화
    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    with tqdm(loader, desc=desc) as loop:
        for texts, masks, labels in loop:

            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            optimizer.zero_grad()

            #순전파
            outputs = model(texts, attention_mask=masks, labels=labels)  # model이 반환하는 값이 (loss, logits)
            loss = outputs.loss  # loss 추출
            logits = outputs.logits  # logits 추출

            #역전파
            loss.backward()
            optimizer.step()

            #정확도 계산
            batch_accuracy = accuracy_sum(logits, labels)
            train_accuracy += batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=train_accuracy / train_epoch_size)

    return {
        "train/accuracy": train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss
    }


def validate(model: nn.Module, device: str, loader: DataLoader, votes=1, desc='Validation'):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}', disable=False)]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm(records, desc=desc, disable=False) as loop, torch.no_grad():
        for example in loop:
            losses = []
            logit_votes = []

            for texts, masks, labels in example:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                batch_size = texts.shape[0]

                #loss, logits = model(texts, attention_mask=masks, labels=labels)

                outputs = model(texts, attention_mask=masks, labels=labels)  # model이 반환하는 값이 (loss, logits)
                loss = outputs.loss  # loss 추출
                logits = outputs.logits  # logits 추출

                if isinstance(loss, str):
                    raise ValueError(f"Loss가 문자열입니다: {loss}")

                losses.append(loss)
                logit_votes.append(logits)


            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)

            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)

    return {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss
    }

def run(max_epochs=None,
        device=None,
        batch_size=24,
        max_sequence_length=128,
        random_sequence_length=False,
        epoch_size=None,
        seed=None,
        data_dir='data',
        real_dataset='webtext',
        fake_dataset='xl-1542M-k40',
        token_dropout=None,
        large=False,
        learning_rate=2e-5,
        weight_decay=0,
        patience=5,
        model_path = None,
        **kwargs):
    args = locals()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('device:', device)

    model_name = 'roberta-large' if large else 'roberta-base'
    print(model_name)
    tokenization_utils.logger.setLevel('ERROR')
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
    
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        model.train()

    summary(model)

    train_loader, validation_loader = load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                                                    max_sequence_length, random_sequence_length, epoch_size,
                                                    token_dropout, seed)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)

    logdir = os.environ.get("OPENAI_LOGDIR", "logs")
    os.makedirs(logdir, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(logdir) if device == 'cuda' else None

    best_validation_accuracy = 0
    patience_counter = 0
    previous_validation_loss = None

    log_data = []
    excel_path = os.path.join(logdir, "training_logs.xlsx")

    for epoch in epoch_loop:
        train_metrics = train(model, optimizer, device, train_loader, f'Epoch {epoch}')
        validation_metrics = validate(model, device, validation_loader)

        combined_metrics = {**validation_metrics, **train_metrics}

        combined_metrics["train/accuracy"] /= combined_metrics["train/epoch_size"]
        combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
        combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
        combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]

        log_data.append({
        "epoch": epoch,
        "train/accuracy": combined_metrics["train/accuracy"],
        "train/loss": combined_metrics["train/loss"],
        "validation/accuracy": combined_metrics["validation/accuracy"],
        "validation/loss": combined_metrics["validation/loss"],
        })
        
        df = pd.DataFrame(log_data) 
        df.to_excel(excel_path, index=False)

        if device == 'cuda':
            for key, value in combined_metrics.items():
                writer.add_scalar(key, value, global_step=epoch)
            
            if previous_validation_loss is not None and combined_metrics["validation/loss"] > previous_validation_loss:
                print(f"📉 No improvement in validation loss.")
                patience_counter += 1
            elif previous_validation_loss is not None and combined_metrics["validation/loss"] < previous_validation_loss:
                patience_counter = 0
            previous_validation_loss = combined_metrics["validation/loss"] 
     

            if combined_metrics["validation/accuracy"] > best_validation_accuracy:
                best_validation_accuracy = combined_metrics["validation/accuracy"]

                torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        args=args
                    ),
                    os.path.join(logdir, "best-model.pt")
                ) 

            if epoch % 10 == 0:
                torch.save(dict(
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    args=args
                ), os.path.join(logdir, f"checkpoint-epoch-{epoch}.pt"))

        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch}")
            break   

    print(f"✅ Training logs saved to {excel_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--epoch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--real-dataset', type=str, default='webtext')
    parser.add_argument('--fake-dataset', type=str, default='xl-1542M-k40')
    parser.add_argument('--token-dropout', type=float, default=None)

    parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()

    run(**vars(args))