"""Training helpers for deceptive product-review detection.

Compares practical word-TF-IDF baselines. A validation split selects the model;
the final test split remains untouched. A word+character feature builder is retained
for optional future experiments, but is not part of the fast default training run.
"""
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "review_model.joblib"
RESULTS_PATH = ROOT / "outputs" / "evaluation_results.csv"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


def word_vectorizer():
    return TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2,
        max_features=8000, sublinear_tf=True, strip_accents="unicode"
    )


def word_char_features():
    return FeatureUnion([
        ("word", word_vectorizer()),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=4000, sublinear_tf=True, strip_accents="unicode"
        )),
    ])


def build_experiments():
    """Return fast, comparable word-TF-IDF baselines for the desktop application."""
    return [
        ("Logistic Regression - Word TF-IDF",
         Pipeline([("tfidf", word_vectorizer()),
                   ("classifier", LogisticRegression(C=1.0, max_iter=1000,
                                                     class_weight="balanced", random_state=42))])),
        ("Multinomial Naive Bayes - Word TF-IDF",
         Pipeline([("tfidf", word_vectorizer()), ("classifier", MultinomialNB())])),
        ("Linear SVM - Word TF-IDF",
         Pipeline([("tfidf", word_vectorizer()),
                   ("classifier", LinearSVC(C=1.0, class_weight="balanced", random_state=42))])),
    ]


def train_and_evaluate(df):
    """Select using a validation split, evaluate once on held-out test data, then refit on all rows."""
    X = df["review_text"].astype(str).str.strip()
    y = df["label"].astype(int)
    keep = X.ne("") & y.isin([0, 1])
    X, y = X[keep], y[keep]
    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 5:
        raise ValueError("Each class needs at least five reviews for train/validation/test evaluation.")

    # Separate final test set first. The test set is not used for model selection.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=43, stratify=y_dev
    )

    rows = []
    fitted = {}
    for name, pipeline in build_experiments():
        pipeline.fit(X_train, y_train)
        # Select a decision threshold using validation data only when the model
        # exposes probabilities. Never tune the threshold on the final test set.
        threshold = 0.5
        if hasattr(pipeline, "predict_proba"):
            val_prob = pipeline.predict_proba(X_val)[:, list(pipeline.classes_).index(1)]
            candidates = [round(float(t), 2) for t in __import__("numpy").arange(0.30, 0.701, 0.02)]
            scored = []
            for candidate in candidates:
                candidate_pred = (val_prob >= candidate).astype(int)
                macro = float(f1_score(y_val, candidate_pred, average="macro", zero_division=0))
                scored.append((macro, candidate))
            val_macro_f1, threshold = max(scored, key=lambda item: item[0])
            val_pred = (val_prob >= threshold).astype(int)
            pipeline.decision_threshold_ = float(threshold)
        else:
            val_pred = pipeline.predict(X_val)
            val_macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        fitted[name] = (pipeline, val_macro_f1)
        rows.append({
            "Model": name,
            "Validation Macro F1": val_macro_f1,
            "Validation Accuracy": float(accuracy_score(y_val, val_pred)),
            "Validation Deceptive Precision": float(precision_score(y_val, val_pred, pos_label=1, zero_division=0)),
            "Validation Deceptive Recall": float(recall_score(y_val, val_pred, pos_label=1, zero_division=0)),
            "Validation Deceptive F1": float(f1_score(y_val, val_pred, pos_label=1, zero_division=0)),
            "Decision Threshold": float(threshold),
            "Validation Records": int(len(y_val)),
        })

    best_name, (best_pipeline, best_validation_f1) = max(
        fitted.items(), key=lambda item: item[1][1]
    )
    # Final test is evaluated only after the validation-based selection is complete.
    threshold = float(getattr(best_pipeline, "decision_threshold_", 0.5))
    if hasattr(best_pipeline, "predict_proba"):
        test_prob = best_pipeline.predict_proba(X_test)[:, list(best_pipeline.classes_).index(1)]
        y_pred = (test_prob >= threshold).astype(int)
    else:
        y_pred = best_pipeline.predict(X_test)
    metrics = {
        "decision_threshold": threshold,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["Genuine", "Deceptive"],
            zero_division=0
        ),
        "test_records": int(len(y_test)),
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "total_records": int(len(X)),
        "best_model": best_name,
        "best_validation_macro_f1": best_validation_f1,
    }
    results_df = pd.DataFrame(rows).sort_values("Validation Macro F1", ascending=False)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)

    # Refit a fresh copy on all labeled rows for the deployed predictor. Reported
    # metrics above remain from the model evaluated on the untouched test split.
    production_pipeline = clone(best_pipeline)
    production_pipeline.fit(X, y)
    production_pipeline.decision_threshold_ = threshold
    joblib.dump(production_pipeline, MODEL_PATH)
    results_df.to_csv(RESULTS_PATH, index=False)
    metadata = {k: v for k, v in metrics.items()
                if k not in ("confusion_matrix", "classification_report")}
    metadata["label_mapping"] = {"0": "Genuine / truthful", "1": "Deceptive / suspicious"}
    metadata["decision_threshold"] = threshold
    metadata["threshold_selection"] = "Selected on validation data by macro F1; not tuned on final test set"
    metadata["deployment_model_refit_on_all_labeled_rows"] = True
    metadata["test_metrics_source"] = "Selected model evaluated on held-out test data before final refit"
    metadata["selection_method"] = "Validation macro F1; final test split not used for selection"
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics["results_df"] = results_df
    metrics["y_test"] = y_test
    metrics["y_pred"] = y_pred
    return production_pipeline, metrics

