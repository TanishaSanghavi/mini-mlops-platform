from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/raw/creditcard.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")


def run_ingestion() -> None:
    """Load raw credit card data, split 80/20, and save processed train/test sets."""
    if not RAW_DATA_PATH.is_file():
        print(f"Error: raw dataset not found at {RAW_DATA_PATH.resolve()}")
        print("Place creditcard.csv in data/raw/ and run again.")
        return

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print("Fraud class distribution:")
    print(df["Class"].value_counts().sort_index())

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Saved train set to {TRAIN_PATH} (shape: {train_df.shape})")
    print(f"Saved test set to {TEST_PATH} (shape: {test_df.shape})")


if __name__ == "__main__":
    run_ingestion()
