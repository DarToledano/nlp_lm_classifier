from data.language_model_dataset import TextDataset
from torch.utils.data import DataLoader
from collections import Counter
from data.config import SEQ_LEN, BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, EPOCHS, DEVICE

def build_vocab(reviews, min_freq=2):
    counter = Counter(token for review in reviews for token in review)
    vocab = {"<UNK>": 0}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    print("\n len(vocab): ", len(vocab))
    return vocab

def create_datasets_and_loaders(train_data, val_data, test_data, vocab):
    train_dataset = TextDataset(train_data, vocab, seq_len=SEQ_LEN)
    val_dataset = TextDataset(val_data, vocab, seq_len=SEQ_LEN)
    test_dataset = TextDataset(test_data, vocab, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader

from data.classification_dataset import ClassificationDataset
from torch.utils.data import DataLoader

def create_classification_datasets_and_loaders(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vocab):
    train_dataset = ClassificationDataset(train_texts, train_labels, vocab, seq_len=SEQ_LEN)
    val_dataset = ClassificationDataset(val_texts, val_labels, vocab, seq_len=SEQ_LEN)
    test_dataset = ClassificationDataset(test_texts, test_labels, vocab, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader
