from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create compact house-price features used by the baseline models.

    The transforms stay intentionally simple: age-based features, total areas,
    and binary indicators for common quality signals.
    """
    out = df.copy()

    # Age and recency features often capture more signal than raw build years.
    out["HouseAge"] = out["YrSold"] - out["YearBuilt"]
    out["RemodAge"] = out["YrSold"] - out["YearRemodAdd"]
    # Combine area-related columns into a few stable summary features.
    out["TotalSF"] = out["TotalBsmtSF"].fillna(0) + out["1stFlrSF"].fillna(0) + out["2ndFlrSF"].fillna(0)
    out["TotalBath"] = (
        out["FullBath"].fillna(0)
        + 0.5 * out["HalfBath"].fillna(0)
        + out["BsmtFullBath"].fillna(0)
        + 0.5 * out["BsmtHalfBath"].fillna(0)
    )
    out["PorchSF"] = (
        out["OpenPorchSF"].fillna(0)
        + out["EnclosedPorch"].fillna(0)
        + out["3SsnPorch"].fillna(0)
        + out["ScreenPorch"].fillna(0)
    )
    out["HasPool"] = (out["PoolArea"].fillna(0) > 0).astype(int)
    out["HasGarage"] = out["GarageArea"].fillna(0).gt(0).astype(int)
    out["HasBasement"] = out["TotalBsmtSF"].fillna(0).gt(0).astype(int)
    out["IsRemodeled"] = (out["YearBuilt"] != out["YearRemodAdd"]).astype(int)
    out["LogLotArea"] = np.log1p(out["LotArea"].fillna(0))

    # Drop identifier before model fitting to avoid leaking row identity.
    out = out.drop(columns=["Id"], errors="ignore")
    return out


def split_target(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the training frame into feature matrix and log-transformed target."""
    y = np.log1p(train_df["SalePrice"]).astype(float)
    X = train_df.drop(columns=["SalePrice"])
    return X, y