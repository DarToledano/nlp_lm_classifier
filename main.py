from data.preprocess import (
    load_imdb_dataset_with_labels,
    plot_review_length_histogram,
    plot_top_tokens,
    print_rare_token_stats,
    print_length_statistics
)
from train import train_pipeline
from evaluate import run_evaluation_language_model,run_evaluation_classification
from experiments.experiment_a import run_experiment_a
from experiments.experiment_b import run_experiment_b
def perform_eda(train_data, val_data, test_data):
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


if __name__ == "__main__":

    # Task 1
    print("==> Loading raw dataset for EDA...")
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_imdb_dataset_with_labels("aclImdb")
    perform_eda(train_data, val_data, test_data)
    print("\n EDA complete.")

    print("\n==> Starting training pipeline...")
    vocab = train_pipeline(train_data, val_data, test_data)
    print("\n Full pipeline complete.")

    print("\n==> Starting evaluation ...")
    run_evaluation_language_model(train_data, test_data)
    print("\n==> Evaluation complete.")

    # Task 2
    print("\n==> Starting classification ...")

    print("\n==> Starting experiment A:")
    preds_a, test_y = run_experiment_a(train_data, train_labels, val_data, val_labels, test_data, test_labels)
    print("\n==> tarting evaluation experiment A:")
    run_evaluation_classification(preds_a, test_y, test_labels)

    print("\n==> Running Experiment B:")
    model = run_experiment_b(train_data, train_labels, val_data, val_labels, test_data, test_labels)
    print("\n==> Evaluation for Experiment B:")




