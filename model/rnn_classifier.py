import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, embedding_weights, freeze_embeddings=True):
        super(RNNClassifier, self).__init__()

        # Embedding layer with pre-trained embeddings (Word2Vec)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_weights)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # RNN layer (you can change to LSTM or GRU if you want)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Fully connected output layer for classification (binary: 0 or 1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        output, hidden = self.rnn(embedded)  # hidden: [1, batch_size, hidden_dim]
        final_hidden = hidden[-1]  # Take the last layer hidden state: [batch_size, hidden_dim]
        logits = self.fc(final_hidden)  # [batch_size, num_classes]
        return logits
