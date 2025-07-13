import torch.nn as nn
from data.language_model_dataset import TextDataset
from model.language_model import LanguageModel
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from data.config import SEQ_LEN, BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, EPOCHS, DEVICE
from data.preprocess import load_imdb_dataset_with_labels
from data.data_loader_utils import build_vocab
def plot_model_grpahs(train_losses, val_losses, train_accs, val_accs, train_ppls, val_ppls):
    plt.figure(figsize=(15, 4))

    # Plot Loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png", dpi=300)
    plt.show()

    # Plot Accuracy
    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_plot.png", dpi=300)
    plt.show()

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

def run_evaluation_language_model(vocab,test_data, train_losses, val_losses, train_accs, val_accs, train_ppl, val_ppl):

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

    model.load_state_dict(torch.load("model/model.pth"))
    criterion = nn.CrossEntropyLoss()

    print("==> Evaluating...")
    avg_loss, perplexity = evaluate_model(model, test_loader, criterion)

    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.2f}")

    plot_model_grpahs(train_losses, val_losses, train_accs, val_accs, train_ppl, val_ppl)


def plot_confusion_matrix(filename, preds, true_labels, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neg", "pos"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def run_error_analysis(preds, true_labels, texts, num_examples=10):
    print("\nError Analysis (First few wrong predictions):")
    errors = (preds != true_labels).nonzero(as_tuple=True)[0]
    for idx in errors[:num_examples]:
        print(f"\nReview: {' '.join(texts[idx][:30])}...")
        print(f"True Label: {true_labels[idx].item()} | Predicted: {preds[idx].item()}")

def run_evaluation_classification(experiment, preds, test_y, test_data):

    if experiment == "A":
        filename_cm = "confusion_matrix_A.png"
    else:
        filename_cm = "confusion_matrix_B.png"

    plot_confusion_matrix(filename_cm,preds.cpu(), test_y)
    run_error_analysis(preds.cpu(), test_y, test_data)

