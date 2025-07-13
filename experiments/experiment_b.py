import pickle
import torch
import numpy as np
from gensim.models import KeyedVectors
from model.rnn_classifier import RNNClassifier
from data.data_loader_utils import build_vocab, create_classification_datasets_and_loaders
from data.config import SEQ_LEN, BATCH_SIZE, DEVICE
import matplotlib.pyplot as plt
from evaluate import run_evaluation_classification
from train import train_classifier_B
# Load Word2Vec once at module level
word2vec_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

# Experiment B: Using RNN (LSTM) classifier with pre-trained Word2Vec embeddings (end-to-end sequence classification)

def load_word2vec(vocab, embedding_dim):
    """
    Create an embedding matrix for the given vocab using Word2Vec.
    """
    embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))

    for word, idx in vocab.items():
        if word in word2vec_model:
            embedding_matrix[idx] = word2vec_model[word]

    return torch.tensor(embedding_matrix, dtype=torch.float)

def run_experiment_b(train_data, train_labels, val_data, val_labels, test_data, test_labels):
    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocab size for Experiment B: {len(vocab)}")

    # Step 2: Create Dataloaders
    train_loader, val_loader, test_loader = create_classification_datasets_and_loaders(
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels,
        vocab
    )

    # Step 3: Load Word2Vec embedding matrix
    embedding_dim = 300
    embedding_matrix = load_word2vec(vocab, embedding_dim)

    # Step 4: Define RNN classifier with pre-trained embeddings
    model = RNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=128,
        num_classes=2,
        embedding_weights=embedding_matrix,
        freeze_embeddings=True
    ).to(DEVICE)

    # Step 5: Train classifier
    print("\nTraining RNN classifier (Experiment B)...")
    train_losses, val_losses = train_classifier_B(
        model,
        train_loader,
        val_loader,
        epochs=10
    )

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Train vs Validation Loss (Experiment A)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    filename = "Training_graph_B.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    # Step 7: Evaluate on test set
    print("\nEvaluating on Test Set for Experiment B...")
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    final_preds = torch.cat(all_preds)
    final_labels = torch.cat(all_labels)


    # Step 8: Confusion matrix and error analysis
    experiment = "B"
    run_evaluation_classification(experiment,final_preds, final_labels, test_data)

    return final_preds, final_labels
