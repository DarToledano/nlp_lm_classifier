import os
import re
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s\-]", "", text)  # Keep hyphens
    return text

def load_reviews_from_dir(dir_path):
    reviews = []
    for fname in os.listdir(dir_path):
        with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
            text = f.read()
            cleaned = preprocess_text(text)
            tokens = word_tokenize(cleaned)
            reviews.append(tokens)
    return reviews

def load_imdb_dataset_with_labels(base_dir):
    """
    Loads the IMDB dataset and returns:
    - tokenized reviews: train/val/test
    - labels (0 = negative, 1 = positive)
    """

    train_pos = load_reviews_from_dir(os.path.join(base_dir, "train/pos"))
    train_neg = load_reviews_from_dir(os.path.join(base_dir, "train/neg"))
    test_pos = load_reviews_from_dir(os.path.join(base_dir, "test/pos"))
    test_neg = load_reviews_from_dir(os.path.join(base_dir, "test/neg"))

    train_texts = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)

    test_texts = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)

    combined = list(zip(train_texts, train_labels))
    random.shuffle(combined)
    train_texts, train_labels = zip(*combined)

    split_idx = int(len(train_texts) * 0.9)
    train_texts, val_texts = train_texts[:split_idx], train_texts[split_idx:]
    train_labels, val_labels = train_labels[:split_idx], train_labels[split_idx:]

    return (
        list(train_texts), list(train_labels),
        list(val_texts), list(val_labels),
        test_texts, test_labels
    )

def plot_review_length_histogram(reviews, title):
    lengths = [len(r) for r in reviews]
    plt.hist(lengths, bins=50)
    plt.title(title)
    plt.xlabel("Review Length (tokens)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_top_tokens(reviews, top_n=30):
    all_tokens = [token for review in reviews for token in review]
    counter = Counter(all_tokens)
    most_common = counter.most_common(top_n)

    words, counts = zip(*most_common)
    plt.figure(figsize=(12, 5))
    plt.bar(words, counts)
    plt.xticks(rotation=45)
    plt.title(f"Top {top_n} Most Frequent Tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_rare_token_stats(reviews, cutoff=1):
    all_tokens = [token for review in reviews for token in review]
    counter = Counter(all_tokens)
    total_vocab = len(counter)
    rare_tokens = [token for token, freq in counter.items() if freq <= cutoff]

    print("Rare Token Stats:")
    print(f"  - Total unique tokens: {total_vocab}")
    print(f"  - Tokens occurring ≤ {cutoff} times: {len(rare_tokens)}")
    print(f"  - % of vocabulary that is rare: {len(rare_tokens)/total_vocab:.2%}")

def print_length_statistics(reviews):
    lengths = [len(r) for r in reviews]
    print("\nReview Length Stats:")
    print(f"  - Mean: {np.mean(lengths):.2f}")
    print(f"  - Median: {np.median(lengths):.2f}")
    print(f"  - 90th percentile: {np.percentile(lengths, 90)}")
    print(f"  - Max: {np.max(lengths)}")
    print(f"  - Min: {np.min(lengths)}")
