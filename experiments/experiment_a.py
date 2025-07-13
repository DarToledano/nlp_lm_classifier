import random
import torch
import pickle
import matplotlib.pyplot as plt
from data.data_loader_utils import build_vocab, create_classification_datasets_and_loaders
from model.language_model import LanguageModel
from model.mlp_classifier import MLPClassifier
from data.config import SEQ_LEN, BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, DEVICE
from train import train_classifier_A
from evaluate import run_evaluation_classification

# Experiment A: Using pre-trained Language Model (LM) as a feature extractor (sentence embeddings) + MLP classifier

def extract_sentence_embeddings(dataloader, language_model, device):
    embeddings = []
    labels = []

    language_model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            _, hidden = language_model(batch_inputs)  # Get hidden state from LM
            sentence_embed = hidden[0][-1]  # Take last layer hidden state
            embeddings.append(sentence_embed.cpu())
            labels.append(batch_labels.cpu())

    all_embeddings = torch.cat(embeddings, dim=0)
    all_labels = torch.cat(labels, dim=0)
    return all_embeddings, all_labels


def run_experiment_a(train_data, train_labels, val_data, val_labels, test_data, test_labels):
    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    print(f"Vocab size for Experiment A: {len(vocab)}")

    # Step 1: Reduce training data to 20%
    reduced_size = int(len(train_data) * 0.2)
    combined = list(zip(train_data, train_labels))
    random.shuffle(combined)
    train_data, train_labels = zip(*combined[:reduced_size])

    # Step 3: Create Dataloaders
    train_loader, val_loader, test_loader = create_classification_datasets_and_loaders(
        list(train_data), list(train_labels),
        val_data, val_labels,
        test_data, test_labels,
        vocab
    )

    print(f"Train size (20%): {len(train_loader.dataset)} samples")

    # Step 4: Load pre-trained language model (from task 1)
    language_model = LanguageModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        # dropout=DROPOUT
    )
    language_model.load_state_dict(torch.load("model/model.pth", map_location=torch.device('cpu')))
    language_model.to(DEVICE)

    # Step 5: Extract sentence embeddings from LM
    print("Extracting sentence embeddings for Experiment A...")
    train_embeddings, train_y = extract_sentence_embeddings(train_loader, language_model, DEVICE)
    val_embeddings, val_y = extract_sentence_embeddings(val_loader, language_model, DEVICE)
    test_embeddings, test_y = extract_sentence_embeddings(test_loader, language_model, DEVICE)

    # Step 6: Train MLP classifier
    print("\nTraining MLP classifier on embeddings...")
    input_dim = train_embeddings.shape[1]
    print("\nInput DIM: ", input_dim)

    mlp_model = MLPClassifier(input_dim=input_dim).to(DEVICE)

    train_losses, val_losses = train_classifier_A(
        mlp_model,
        train_embeddings, train_y,
        val_embeddings, val_y,
        epochs=10
    )

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Train vs Validation Loss (Experiment A)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    filename = "Training_graph_A.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    # Step 7: Evaluate on test set
    mlp_model.eval()
    with torch.no_grad():
        preds = mlp_model(test_embeddings.to(DEVICE)).argmax(dim=1)
        acc = (preds.cpu() == test_y).float().mean().item()
        print(f"\nTest Accuracy (Experiment A): {acc:.4f}")

    # Step 9: Confusion matrix and error analysis
    experiment = "A"
    plot_experiments_grpahs(experiment,train_losses,val_losses,)
    run_evaluation_classification(experiment, preds.cpu(), test_y, test_data)

