from data.preprocess import (
    load_imdb_dataset,
    plot_review_length_histogram,
    plot_top_tokens,
    print_rare_token_stats,
    print_length_statistics
)
from data.language_model_dataset import TextDataset
from collections import Counter

if __name__ == "__main__":
    print("Loading IMDB dataset...")
    train_data, val_data, test_data = load_imdb_dataset("aclImdb")

    print("\nDataset loaded:")
    print(f"  - Train: {len(train_data)}")
    print(f"  - Validation: {len(val_data)}")
    print(f"  - Test: {len(test_data)}")

    print("\nSample review (first 20 tokens):")
    print(train_data[0][:20])

    print("\nHistogram of review lengths...")
    plot_review_length_histogram(train_data, "Train Review Lengths")
    plot_review_length_histogram(val_data, "Validation Review Lengths")
    plot_review_length_histogram(test_data, "Test Review Lengths")

    print_length_statistics(train_data)

    print("\nTop 30 most frequent tokens...")
    plot_top_tokens(train_data, top_n=30)

    print("\nRare token analysis...")
    print_rare_token_stats(train_data, cutoff=1)

    print("\nEDA complete.")

    # 👉 Add this block here
    def build_vocab(reviews, min_freq=2):
        counter = Counter(token for review in reviews for token in review)
        vocab = {"<UNK>": 0}
        for token, freq in counter.items():
            if freq >= min_freq:
                vocab[token] = len(vocab)
        return vocab

    print("Building vocabulary from training data...")
    vocab = build_vocab(train_data)
    print(f"Vocabulary size: {len(vocab)}")

    print("Creating TextDataset...")
    train_dataset = TextDataset(train_data, vocab, seq_len=30)

    print(f"Total training samples: {len(train_dataset)}")


