"""
Data gathering utilities for the WOS-46985 (Web of Science) dataset.

Handles:
  - downloading the raw Mendeley zip (with a manual-upload fallback path)
  - recursively unzipping (the Mendeley archive is nested: outer zip -> WebOfScience.zip -> per-variant folders)
  - parsing X.txt / Y.txt / YL1.txt into a pandas DataFrame
  - building a fixed, stratified train/val/test split (saved to disk so every method/model
    trains on identical data)

Every function here is idempotent: safe to call repeatedly, will skip work already done.
"""

import os
import zipfile
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

MENDELEY_ZIP_URL = (
    "https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/"
    "c9ea673d-5542-44c0-ab7b-f1311f7d61df/file_downloaded"
)

RAW_DIR = "data/wos_raw"
SPLITS_DIR = "data/wos_splits"


def _find_wos_dir(raw_dir: str) -> str | None:
    for root, _, files in os.walk(raw_dir):
        if "X.txt" in files and "46985" in root:
            return root
    return None


def _unzip_recursive(zip_path: str, extract_dir: str, seen: set | None = None) -> None:
    if seen is None:
        seen = set()
    if zip_path in seen:
        return
    seen.add(zip_path)
    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            if fn.endswith(".zip"):
                fp = os.path.join(root, fn)
                if fp not in seen:
                    _unzip_recursive(fp, extract_dir, seen)


def download_and_extract(raw_dir: str = RAW_DIR, manual_zip_path: str | None = None) -> str:
    """
    Downloads (or uses a manually-provided zip of) the WOS dataset and extracts it.
    Returns the path to the folder containing X.txt / Y.txt / YL1.txt for the WOS46985 variant.
    """
    os.makedirs(raw_dir, exist_ok=True)

    existing = _find_wos_dir(raw_dir)
    if existing:
        print(f"Data already present at: {existing} — skipping download.")
        return existing

    if manual_zip_path and os.path.exists(manual_zip_path):
        outer_zip = manual_zip_path
    else:
        print("Downloading from Mendeley...")
        resp = requests.get(MENDELEY_ZIP_URL, timeout=120)
        print("Status:", resp.status_code)
        if resp.status_code != 200:
            raise RuntimeError(
                "Mendeley download failed (link may have expired). "
                "Download manually from https://data.mendeley.com/datasets/9rw3vkcfy4/6 "
                "and re-run with manual_zip_path='<path to your downloaded zip>'."
            )
        outer_zip = os.path.join(raw_dir, "outer.zip")
        with open(outer_zip, "wb") as f:
            f.write(resp.content)

    _unzip_recursive(outer_zip, raw_dir)

    wos_dir = _find_wos_dir(raw_dir)
    if wos_dir is None:
        raise RuntimeError(f"Could not locate WOS46985 folder under {raw_dir} after extraction.")
    return wos_dir


def build_dataframe(wos_dir: str) -> pd.DataFrame:
    def read_lines(fp):
        with open(fp, encoding="utf-8", errors="ignore") as f:
            return [line.strip() for line in f]

    texts = read_lines(os.path.join(wos_dir, "X.txt"))
    y_fine = read_lines(os.path.join(wos_dir, "Y.txt"))
    yl1 = read_lines(os.path.join(wos_dir, "YL1.txt"))

    return pd.DataFrame({
        "text": texts,
        "label_fine": [int(v) for v in y_fine],
        "label_domain": [int(v) for v in yl1],
    })


def get_splits(splits_dir: str = SPLITS_DIR, manual_zip_path: str | None = None,
                force_rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main entrypoint. Returns (train_df, val_df, test_df).
    Reuses saved CSVs if they already exist, unless force_rebuild=True.
    """
    train_fp = os.path.join(splits_dir, "train.csv")
    val_fp = os.path.join(splits_dir, "val.csv")
    test_fp = os.path.join(splits_dir, "test.csv")

    if not force_rebuild and all(os.path.exists(fp) for fp in (train_fp, val_fp, test_fp)):
        print("Loading existing splits from disk.")
        return (
            pd.read_csv(train_fp),
            pd.read_csv(val_fp),
            pd.read_csv(test_fp),
        )

    wos_dir = download_and_extract(manual_zip_path=manual_zip_path)
    wos_df = build_dataframe(wos_dir)
    print("Total examples:", len(wos_df))

    train_df, temp_df = train_test_split(
        wos_df, test_size=0.3, stratify=wos_df["label_domain"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label_domain"], random_state=42
    )

    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(train_fp, index=False)
    val_df.to_csv(val_fp, index=False)
    test_df.to_csv(test_fp, index=False)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Quick standalone check: `python src/data_utils.py`
    tr, va, te = get_splits()
    print(tr.head(3))
