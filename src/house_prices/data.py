from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Kaggle train/test CSV files from the data directory."""
    data_path = Path(data_dir)
    train_df = pd.read_csv(data_path / "train.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    return train_df, test_df