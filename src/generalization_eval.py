"""Evaluate random, product-held-out, and category-held-out generalization.

Run from project root:
    python -m src.generalization_eval

This is a diagnostic evaluation using one fixed word-TF-IDF + Logistic Regression
pipeline. It is separate from the GUI's model-selection experiment.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "amazon_product_reviews_cleaned.csv"
OUT = ROOT / "outputs" / "generalization_results.csv"
DETAIL = ROOT / "outputs" / "generalization_details.json"


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            max_features=25000, sublinear_tf=True, strip_accents="unicode"
        )),
        ("classifier", LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=42
        ))
    ])


def evaluate_split(df, train_idx, test_idx, split_name):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    model = build_model()
    model.fit(train["review_text"], train["label"])
    pred = model.predict(test["review_text"])
    result = {
        "Split": split_name,
        "Train rows": int(len(train)),
        "Test rows": int(len(test)),
        "Accuracy": float(accuracy_score(test["label"], pred)),
        "Deceptive precision": float(precision_score(test["label"], pred, pos_label=1, zero_division=0)),
        "Deceptive recall": float(recall_score(test["label"], pred, pos_label=1, zero_division=0)),
        "Deceptive F1": float(f1_score(test["label"], pred, pos_label=1, zero_division=0)),
        "Macro F1": float(f1_score(test["label"], pred, average="macro", zero_division=0)),
        "Confusion matrix [ [TN, FP], [FN, TP] ]": confusion_matrix(test["label"], pred, labels=[0, 1]).tolist(),
    }
    details = {
        "split": split_name,
        "train_label_counts": train["label"].value_counts().sort_index().to_dict(),
        "test_label_counts": test["label"].value_counts().sort_index().to_dict(),
        "confusion_matrix_labels_0_1": result["Confusion matrix [ [TN, FP], [FN, TP] ]"],
    }
    if "product_category" in df.columns:
        details["test_categories"] = sorted(test["product_category"].dropna().astype(str).unique().tolist())
    if "product_id" in df.columns:
        details["train_unique_product_ids"] = int(train["product_id"].nunique())
        details["test_unique_product_ids"] = int(test["product_id"].nunique())
        details["product_id_overlap"] = int(
            len(set(train["product_id"].dropna().astype(str)) &
                set(test["product_id"].dropna().astype(str)))
        )
    return result, details


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    needed = {"review_text", "label"}
    if needed - set(df.columns):
        raise ValueError(f"Dataset needs columns: {sorted(needed)}")
    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["review_text"].ne("") & df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("Both labels 0 and 1 are required.")

    results, details = [], []
    indices = list(range(len(df)))

    # 1) Random stratified split: comparable to the original project baseline.
    tr, te = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=df["label"]
    )
    r, d = evaluate_split(df, tr, te, "Random stratified split")
    results.append(r); details.append(d)

    # 2) Product-held-out split: no product ID should occur on both sides.
    if "product_id" in df.columns and df["product_id"].notna().any():
        groups = df["product_id"].fillna("__MISSING_PRODUCT_ID__").astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr, te = next(splitter.split(df, df["label"], groups=groups))
        if df.iloc[tr]["label"].nunique() == 2 and df.iloc[te]["label"].nunique() == 2:
            r, d = evaluate_split(df, tr, te, "Product-held-out split")
            results.append(r); details.append(d)
        else:
            print("Skipped product-held-out split: one partition lacks a class. Try another random_state.")

    # 3) Category-held-out split: test only on categories excluded from training.
    if "product_category" in df.columns and df["product_category"].notna().any():
        groups = df["product_category"].fillna("__MISSING_CATEGORY__").astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr, te = next(splitter.split(df, df["label"], groups=groups))
        if df.iloc[tr]["label"].nunique() == 2 and df.iloc[te]["label"].nunique() == 2:
            r, d = evaluate_split(df, tr, te, "Category-held-out split")
            results.append(r); details.append(d)
        else:
            print("Skipped category-held-out split: one partition lacks a class. Try another random_state.")

    result_df = pd.DataFrame(results)
    OUT.parent.mkdir(exist_ok=True)
    result_df.to_csv(OUT, index=False)
    DETAIL.write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
    print(result_df[["Split", "Train rows", "Test rows", "Accuracy",
                     "Deceptive precision", "Deceptive recall", "Deceptive F1", "Macro F1"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {OUT}")
    print(f"Saved details: {DETAIL}")
    print("\nInterpretation: random split measures in-distribution performance; product-held-out tests unseen products; category-held-out tests transfer to unseen categories.")


if __name__ == "__main__":
    main()
