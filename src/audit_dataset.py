"""Audit the labeled Amazon review CSV for duplicates, label balance, and metadata shortcuts.

Run from the project root:
    python -m src.audit_dataset
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "amazon_product_reviews_cleaned.csv"
OUT = ROOT / "outputs" / "dataset_audit.txt"


def normalized_text(series):
    return (series.fillna("").astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    required = {"review_text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    norm = normalized_text(df["review_text"])
    nonempty = norm.ne("")
    df = df.loc[nonempty].copy()
    norm = norm.loc[nonempty]

    lines = []
    def add(s=""):
        lines.append(str(s))

    add("SPAM REVIEW DETECTION — DATASET AUDIT")
    add("=" * 55)
    add(f"Dataset file: {DATA.name}")
    add(f"Rows with valid binary labels and non-empty text: {len(df):,}")
    add(f"Columns: {', '.join(df.columns)}")
    add()
    add("1. LABEL BALANCE")
    add(df["label"].map({0: "Genuine (0)", 1: "Deceptive (1)"}).value_counts().to_string())
    add()
    add("2. DUPLICATE REVIEW TEXT")
    add(f"Exact duplicate review_text rows beyond first: {df['review_text'].duplicated().sum():,}")
    add(f"Duplicate after lowercase + whitespace normalization: {normalized_text(df['review_text']).duplicated().sum():,}")
    conflict = df.assign(_norm=normalized_text(df["review_text"])).groupby("_norm")["label"].nunique()
    add(f"Normalized text strings that have conflicting labels: {(conflict > 1).sum():,}")
    add()
    add("3. METADATA COVERAGE")
    for col in ("product_category", "product_id", "product_title", "rating", "verified_purchase", "doc_id"):
        if col in df.columns:
            add(f"{col}: missing={df[col].isna().sum():,}; unique={df[col].nunique(dropna=True):,}")
    if "product_id" in df.columns:
        ids = df["product_id"].dropna().astype(str)
        add(f"Rows with a product_id appearing more than once: {df['product_id'].astype(str).duplicated(keep=False).sum():,}")
        add(f"Unique product IDs: {ids.nunique():,}")
    add()
    add("4. LABEL RATE BY PRODUCT CATEGORY")
    if "product_category" in df.columns:
        cat = df.groupby("product_category").agg(
            reviews=("label", "size"),
            genuine=("label", lambda s: int((s == 0).sum())),
            deceptive=("label", lambda s: int((s == 1).sum())),
            deceptive_rate=("label", "mean")
        ).sort_values("deceptive_rate")
        add(cat.to_string(float_format=lambda x: f"{x:.3f}"))
    else:
        add("No product_category column.")
    add()
    add("5. LABEL ASSOCIATION WITH METADATA (DIAGNOSTIC ONLY)")
    for col in ("rating", "verified_purchase"):
        if col in df.columns:
            add(f"\n{col}:")
            add(pd.crosstab(df[col].fillna("MISSING"), df["label"], normalize="index")
                .rename(columns={0: "genuine_fraction", 1: "deceptive_fraction"})
                .round(3).to_string())
    add()
    add("6. INTERPRETATION")
    add("- Exact duplicate text should not appear in both train and test partitions.")
    add("- A product-held-out split tests performance on products not seen during training.")
    add("- A category-held-out split is a harder test of transfer to unseen product categories.")
    add("- Strong metadata/label association is a warning to investigate source construction; it is not proof of bad labels.")
    add("- Rating and verified-purchase fields are NOT used as model input by the text-only classifier.")
    add("- Dataset labels are inherited from the source corpus and are not independently verified by this audit.")
    report = "\n".join(lines) + "\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved audit report: {OUT}")


if __name__ == "__main__":
    main()
