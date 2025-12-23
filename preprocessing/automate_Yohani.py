import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# =========================
# TENURE BINNING (SAMA PERSIS DENGAN NOTEBOOK EKSPERIMEN)
# =========================
TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip outliers per-feature using IQR bounds learned from TRAIN only."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.low_ = None
        self.high_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if X.size == 0 or X.ndim != 2 or X.shape[1] == 0:
            self.low_ = None
            self.high_ = None
            return self

        q1 = np.nanquantile(X, 0.25, axis=0)
        q3 = np.nanquantile(X, 0.75, axis=0)
        iqr = q3 - q1

        self.low_ = q1 - self.factor * iqr
        self.high_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.low_ is None or self.high_ is None or X.size == 0:
            return X
        return np.clip(X, self.low_, self.high_)


def add_tenure_binning(df: pd.DataFrame) -> pd.DataFrame:
    """Add tenure_bin exactly like in experiment notebook."""
    if "tenure" not in df.columns:
        return df

    out = df.copy()
    out["tenure_bin"] = pd.cut(out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    out["tenure_bin"] = out["tenure_bin"].astype("category")
    return out


def build_onehot_encoder():
    """Compat for different sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocess_data(
    input_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    # 1) Check file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di path: {input_path}")

    # 2) Load dataset
    df = pd.read_csv(input_path)

    # 3) Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
        .str.lower()
    )

    # 4) Validate required columns
    required_cols = {"churn", "totalcharges"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib hilang: {missing}. Periksa header CSV kamu.")

    # 5) Drop duplicates
    df = df.drop_duplicates()

    # 6) Drop ID column (if exists)
    if "customerid" in df.columns:
        df = df.drop(columns=["customerid"])

    # 7) Convert totalcharges to numeric
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    # 8) Encode target churn with validation
    churn_raw = df["churn"].astype(str).str.strip().str.lower()
    df["churn"] = churn_raw.map({"no": 0, "yes": 1})

    if df["churn"].isnull().any():
        bad_values = churn_raw[df["churn"].isnull()].unique()
        raise ValueError(
            f"Nilai churn tidak terduga (bukan Yes/No): {bad_values}. "
            "Periksa isi kolom churn."
        )

    # 8.1) Add tenure_bin (SAMA dengan eksperimen)
    df = add_tenure_binning(df)

    # 9) Split features and target
    X = df.drop(columns=["churn"])
    y = df["churn"].astype(int)

    # 10) Identify numeric & categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)

    # 11) Train-test split (IQR bounds belajar dari TRAIN)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 12) Pipelines
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("iqr_clip", IQRClipper(factor=1.5)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", build_onehot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    # 13) Transform
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # 13.1) Feature names for CSV header
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_enc.shape[1])]

    # 14) Save outputs
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train_enc, columns=feature_names).to_csv(
        os.path.join(output_dir, "X_train.csv"), index=False
    )
    pd.DataFrame(X_test_enc, columns=feature_names).to_csv(
        os.path.join(output_dir, "X_test.csv"), index=False
    )
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print("Preprocessing automation completed successfully")
    print(f"Saved to: {output_dir}/")
    print(f"Train shape: {X_train_enc.shape} | Test shape: {X_test_enc.shape}")


if __name__ == "__main__":
    preprocess_data(
        input_path="namadataset_raw/Telco_Customer_Churn.csv",
        output_dir="preprocessing/TelcoCustomer_preprocessing",
    )
