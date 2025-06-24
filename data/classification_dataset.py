import torch
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, seq_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        label = self.labels[idx]

        # Convert tokens to indices
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        # Pad or truncate
        if len(indices) < self.seq_len:
            indices += [0] * (self.seq_len - len(indices))  # 0 for padding
        else:
            indices = indices[:self.seq_len]

        return torch.tensor(indices), torch.tensor(label)
