from data.language_model_dataset import TextDataset
from data.preprocess import load_imdb_dataset
from model.language_model import LanguageModel
from torch.utils.data import DataLoader
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Reduced model and training parameters for faster debugging
SEQ_LEN = 30
BATCH_SIZE = 64
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_vocab(reviews, min_freq=2):
    counter = Counter(token for review in reviews for token in review)
    vocab = {"<UNK>": 0}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def create_datasets_and_loaders(train_data, val_data, test_data, vocab):
    train_dataset = TextDataset(train_data, vocab, seq_len=SEQ_LEN)
    val_dataset = TextDataset(val_data, vocab, seq_len=SEQ_LEN)
    test_dataset = TextDataset(test_data, vocab, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return total_loss / len(dataloader), correct / total

def train_pipeline():
    # ✅ Use smaller dataset for quick training
    train_data, val_data, test_data = load_imdb_dataset("aclImdb")

    vocab = build_vocab(train_data)
    train_loader, val_loader, _ = create_datasets_and_loaders(train_data, val_data, test_data, vocab)

    model = LanguageModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    # Optional: comment out for speed
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.show()
