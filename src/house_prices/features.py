from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["HouseAge"] = out["YrSold"] - out["YearBuilt"]
    out["RemodAge"] = out["YrSold"] - out["YearRemodAdd"]
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

    out = out.drop(columns=["Id"], errors="ignore")
    return out


def split_target(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = np.log1p(train_df["SalePrice"]).astype(float)
    X = train_df.drop(columns=["SalePrice"])
    return X, y