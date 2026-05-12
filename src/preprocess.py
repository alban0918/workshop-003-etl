"""
Data loading, cleaning, and feature engineering for the World Happiness dataset (2015-2019).
Produces a unified DataFrame with consistent column names and a 'year' column.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

COLUMN_MAP = {
    # 2015 / 2016
    "Happiness Score": "happiness_score",
    "Economy (GDP per Capita)": "gdp_per_capita",
    "Family": "social_support",
    "Health (Life Expectancy)": "life_expectancy",
    "Freedom": "freedom",
    "Trust (Government Corruption)": "corruption",
    "Generosity": "generosity",
    # 2017
    "Happiness.Score": "happiness_score",
    "Economy..GDP.per.Capita.": "gdp_per_capita",
    "Health..Life.Expectancy.": "life_expectancy",
    "Trust..Government.Corruption.": "corruption",
    # 2018 / 2019
    "Score": "happiness_score",
    "GDP per capita": "gdp_per_capita",
    "Social support": "social_support",
    "Healthy life expectancy": "life_expectancy",
    "Freedom to make life choices": "freedom",
    "Perceptions of corruption": "corruption",
    # country
    "Country": "country",
    "Country or region": "country",
}

FEATURES = ["gdp_per_capita", "social_support", "life_expectancy", "freedom", "generosity", "corruption"]
TARGET = "happiness_score"


def load_year(year: int) -> pd.DataFrame:
    path = DATA_DIR / f"{year}.csv"
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_MAP)
    df["year"] = year
    # Keep only needed columns (some may be missing in a year)
    cols = ["country", "year", TARGET] + FEATURES
    existing = [c for c in cols if c in df.columns]
    return df[existing]


def load_all() -> pd.DataFrame:
    frames = [load_year(y) for y in range(2015, 2020)]
    df = pd.concat(frames, ignore_index=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Fill numeric NaNs with column median
    for col in FEATURES + [TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    # Drop rows still missing target
    df = df.dropna(subset=[TARGET])
    return df


def get_feature_matrix(df: pd.DataFrame):
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].values
    y = df[TARGET].values
    return X, y, available


if __name__ == "__main__":
    df = clean(load_all())
    print(df.shape)
    print(df.describe())
    df.to_csv(Path(__file__).parent.parent / "outputs" / "combined.csv", index=False)
    print("Saved combined.csv")
