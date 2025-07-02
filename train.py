from model.language_model import LanguageModel
import matplotlib.pyplot as plt
from data.data_loader_utils import build_vocab, create_datasets_and_loaders
from data.config import SEQ_LEN, BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, EPOCHS, DEVICE, LEARNING_RATE
import math
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from data.config import DEVICE
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

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, correct / total, perplexity

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

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, correct / total, perplexity

def train_pipeline(train_data, val_data, test_data):

    vocab = build_vocab(train_data)
    with open("data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    train_loader, val_loader, _ = create_datasets_and_loaders(train_data, val_data, test_data, vocab)

    model = LanguageModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        #dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc, train_ppl = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_ppl = validate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val PPL:   {val_ppl:.2f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    torch.save(model.state_dict(), "model/model.pth")
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    return vocab

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

# Task 2
from model.language_model import LanguageModel
import matplotlib.pyplot as plt
from data.data_loader_utils import build_vocab, create_datasets_and_loaders
from data.config import SEQ_LEN, BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, EPOCHS, DEVICE, LEARNING_RATE
import math
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from data.config import DEVICE

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

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, correct / total, perplexity

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

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, correct / total, perplexity

def train_pipeline(train_data, val_data, test_data):

    vocab = build_vocab(train_data)
    with open("data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    train_loader, val_loader, _ = create_datasets_and_loaders(train_data, val_data, test_data, vocab)

    model = LanguageModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        #dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc, train_ppl = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_ppl = validate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val PPL:   {val_ppl:.2f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    torch.save(model.state_dict(), "model/model.pth")
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    return vocab

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

# Task 2
def train_classifier_A(model, train_X, train_y, val_X, val_y, epochs=10, lr=0.001):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    train_X, train_y = train_X.to(DEVICE), train_y.to(DEVICE)
    val_X, val_y = val_X.to(DEVICE), val_y.to(DEVICE)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y).item()
            val_losses.append(val_loss)

            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == val_y).float().mean().item()

        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

    return train_losses, val_losses

def train_classifier_B(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # -------- Train ----------
        model.train()
        total_train_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- Validation ----------
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")

    return train_losses, val_losses

def train_classifier_B(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # -------- Train ----------
        model.train()
        total_train_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- Validation ----------
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")

    return train_losses, val_losses
