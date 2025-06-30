import random
import torch
import numpy as np
from gensim.models import KeyedVectors
from model.rnn_classifier import RNNClassifier
from data.data_loader_utils import build_vocab, create_classification_datasets_and_loaders
from data.config import SEQ_LEN, BATCH_SIZE, DEVICE
from train import train_classifier
import matplotlib.pyplot as plt

word2vec_model = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)

def load_word2vec(vocab, embedding_dim):
    """
    Create an embedding matrix for the given vocab using the loaded Word2Vec model.
    """
    embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))

    for word, idx in vocab.items():
        if word in word2vec_model:
            embedding_matrix[idx] = word2vec_model[word]

    return torch.tensor(embedding_matrix, dtype=torch.float)

def run_experiment_b(train_data, train_labels, val_data, val_labels, test_data, test_labels):
    # Step 1: Build vocab
    vocab = build_vocab(train_data)

    # Step 2: Create DataLoaders
    train_loader, val_loader, test_loader = create_classification_datasets_and_loaders(
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels,
        vocab,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE
    )

    # Step 3: Load Word2Vec embedding matrix
    print("Loading Word2Vec embeddings for vocab...")
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
    train_losses, val_losses = train_classifier(
        model,
        train_loader,
        val_loader,
        epochs=10
    )

    # Step 6: Plot loss curves
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Train vs. Validation Loss (Experiment B)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    return model
