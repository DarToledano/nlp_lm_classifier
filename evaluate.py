# evaluation.py

import torch
import torch.nn as nn
from data.language_model_dataset import TextDataset
from data.preprocess import load_imdb_dataset
from model.language_model import LanguageModel
from torch.utils.data import DataLoader
from collections import Counter
import math

# Config
SEQ_LEN = 30
BATCH_SIZE = 64
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_vocab(reviews, min_freq=2):
    counter = Counter(token for review in reviews for token in review)
    vocab = {"<UNK>": 0}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def create_test_loader(test_data, vocab):
    test_dataset = TextDataset(test_data, vocab, seq_len=SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return test_loader

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def run_evaluation():
    print("==> Loading dataset and vocabulary...")
    train_data, _, test_data = load_imdb_dataset("aclImdb")
    vocab = build_vocab(train_data)

    print("==> Preparing test loader...")
    test_loader = create_test_loader(test_data, vocab)

    print("==> Loading trained model...")
    model = LanguageModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    model.load_state_dict(torch.load("model.pth"))
    criterion = nn.CrossEntropyLoss()

    print("==> Evaluating...")
    avg_loss, perplexity = evaluate_model(model, test_loader, criterion)

    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.2f}")
