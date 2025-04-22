import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from .dataset import Corpus,EncodedDataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                  max_sequence_length, random_sequence_length, epoch_size=None, token_dropout=None, seed=None):

    real_corpus = Corpus(real_dataset, data_dir=data_dir)

    if fake_dataset == "TWO":
        fake_corpora = [Corpus(name, data_dir=data_dir) for name in ['xl-1542M', 'xl-1542M-k40']]
        fake_test = sum([corpus.test for corpus in fake_corpora], [])
    else:
        fake_corpus = Corpus(fake_dataset, data_dir=data_dir)
        fake_test = fake_corpus.test  # fake 데이터셋의 test 데이터

    Sampler = RandomSampler

    test_dataset = EncodedDataset(real_corpus.test, fake_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=Sampler(test_dataset))  # test loader만 반환

    return test_loader

def test(model: nn.Module, device: str, loader: DataLoader, desc='test'):
    model.eval()

    all_preds = []
    all_labels = []

    test_loss = 0
    test_epoch_size = 0
    test_accuracy = 0

    with tqdm(loader, desc=desc, disable=False) as loop, torch.no_grad():
        for texts, masks, labels in loop:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            outputs = model(texts, attention_mask=masks, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            preds = logits.argmax(dim=-1).cpu().numpy()  # 예측값
            labels = labels.cpu().numpy()  # 실제 정답

            # 정확도 계산 (기존 코드 유지)
            batch_accuracy = (preds == labels).sum()
            test_accuracy += batch_accuracy
            test_epoch_size += batch_size
            test_loss += loss.item() * batch_size

            # F1-score를 위해 전체 예측값 저장
            all_preds.extend(preds)
            all_labels.extend(labels)

            loop.set_postfix(loss=loss.item(), acc=test_accuracy / test_epoch_size)

    # Precision, Recall, F1-score 계산
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    # Confusion Matrix 계산
    cm = confusion_matrix(all_labels, all_preds)
    
    # Confusion Matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    
    

    return {
        "test/accuracy": test_accuracy / test_epoch_size,
        "test/loss": test_loss / test_epoch_size,
        "test/precision": precision,
        "test/recall": recall,
        "test/f1_score": f1
    }


def run(model_path, data_dir='data', real_dataset='webtext', fake_dataset='xl-1542M',
             batch_size=24, max_sequence_length=128, device=None, large=False, random_sequence_length=False,):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    model_name = 'roberta-large' if large else 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)

    # 저장된 모델 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    # 테스트 데이터 로드
    test_loader = load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size, max_sequence_length, random_sequence_length)

    # 테스트 실행
    test_metrics = test(model, device, test_loader)
    print("Test Results:", test_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='./logs/best-model.pt')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--real-dataset', type=str, default='webtext')
    parser.add_argument('--fake-dataset', type=str, default='xl-1542M')
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')

    args = parser.parse_args()

    run(**vars(args))
