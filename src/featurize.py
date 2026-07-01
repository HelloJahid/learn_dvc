"""Featurize stage.

Split the prepared table into stratified train/test sets and standard-scale the features.
The scaler is fit on TRAIN ONLY (then applied to test) to avoid data leakage. Writes
data/features/train.csv and data/features/test.csv for the train/evaluate stages.
"""
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IN = "data/prepared/wine.csv"
OUT_DIR = "data/features"


def main():
    params = yaml.safe_load(open("params.yaml"))
    split = params["split"]
    featurize = params["featurize"]

    df = pd.read_csv(IN)
    X = df.drop(columns=["good"])
    y = df["good"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split["test_size"],
        random_state=split["seed"],
        stratify=y,            # keep the good/bad ratio identical in train & test
    )

    if featurize["scaler"] == "standard":
        scaler = StandardScaler()
        # fit on train, transform both -> no test information leaks into the scaler
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    os.makedirs(OUT_DIR, exist_ok=True)
    train = X_train.assign(good=y_train.values)
    test = X_test.assign(good=y_test.values)
    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)
    print(f"featurize: train={train.shape} test={test.shape} scaler={featurize['scaler']}")


if __name__ == "__main__":
    main()
