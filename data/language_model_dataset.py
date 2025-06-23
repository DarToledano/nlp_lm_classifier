import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokenized_reviews, vocab, seq_len=30):
        """
        Args:
            tokenized_reviews: list of list of tokens
            vocab: dictionary mapping token to index
            seq_len: number of tokens in input sequence
        """
        self.data = []
        self.seq_len = seq_len
        self.vocab = vocab

        for review in tokenized_reviews:
            indexed = [vocab.get(token, vocab["<UNK>"]) for token in review]
            for i in range(len(indexed) - seq_len):
                input_seq = indexed[i:i+seq_len]
                target = indexed[i+seq_len]
                self.data.append((input_seq, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target = self.data[idx]
        return torch.tensor(input_seq), torch.tensor(target)
