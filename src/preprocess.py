"""Preprocess stage.

Read the raw, semicolon-separated wine CSV and turn the integer `quality` score into a
binary `good` target (1 if quality >= threshold, else 0). Writes a clean, comma-separated
table to data/prepared/ for the featurize stage to consume.
"""
import os

import pandas as pd
import yaml

RAW = "data/raw/wine.csv"
OUT_DIR = "data/prepared"
OUT = os.path.join(OUT_DIR, "wine.csv")


def main():
    params = yaml.safe_load(open("params.yaml"))["preprocess"]
    threshold = params["quality_threshold"]

    # The raw UCI file is ';'-separated (Lesson 1). Forget sep=";" and every column
    # collapses into a single string column.
    df = pd.read_csv(RAW, sep=";")

    df["good"] = (df["quality"] >= threshold).astype(int)
    df = df.drop(columns=["quality"])

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT, index=False)  # default comma separator from here on
    print(f"preprocess: wrote {OUT}  shape={df.shape}  good_rate={df['good'].mean():.3f}")


if __name__ == "__main__":
    main()
