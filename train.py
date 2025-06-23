# train.py

from data.language_model_dataset import TextDataset
from torch.utils.data import DataLoader
from collections import Counter
from data.preprocess import load_imdb_dataset

SEQ_LEN = 30
BATCH_SIZE = 64

def build_vocab(reviews, min_freq=2):
    counter = Counter(token for review in reviews for token in review)
    vocab = {"<UNK>": 0}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def create_datasets_and_loaders(train_data, val_data, test_data, vocab):
    print("Creating TextDatasets...")
    train_dataset = TextDataset(train_data, vocab, seq_len=SEQ_LEN)
    val_dataset = TextDataset(val_data, vocab, seq_len=SEQ_LEN)
    test_dataset = TextDataset(test_data, vocab, seq_len=SEQ_LEN)

    print("Wrapping datasets with DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader

def prepare_training():
    print("Loading dataset for training...")
    train_data, val_data, test_data = load_imdb_dataset("aclImdb")

    print("Building vocabulary...")
    vocab = build_vocab(train_data)
    print(f"Vocabulary size: {len(vocab)}")

    train_loader, val_loader, test_loader = create_datasets_and_loaders(
        train_data, val_data, test_data, vocab
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, vocab
