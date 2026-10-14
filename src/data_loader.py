"""Load and validate labeled review datasets for Spam Review Detection."""
from pathlib import Path
import pandas as pd

TEXT_ALIASES = ("review_text", "review", "text", "content", "review_content")
LABEL_ALIASES = ("label", "class", "target", "is_fake", "deceptive", "spam")


def _find_column(columns, aliases):
    lookup = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def load_reviews(path):
    """Read CSV and return normalized dataframe with review_text and label (0/1)."""
    path = Path(path)
    # Try common encodings and separators. Most CSV files use UTF-8/comma.
    last_error = None
    df = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
            break
        except Exception as exc:
            last_error = exc
    if df is None:
        raise ValueError(f"Could not read the dataset: {last_error}")
    text_col = _find_column(df.columns, TEXT_ALIASES)
    label_col = _find_column(df.columns, LABEL_ALIASES)
    if text_col is None or label_col is None:
        raise ValueError(
            "Dataset must have a review-text column (review_text/review/text/content) "
            "and a label column (label/class/target). Rename the columns or use the template."
        )

    out = df[[text_col, label_col]].copy()
    out.columns = ["review_text", "label"]
    out["review_text"] = out["review_text"].fillna("").astype(str).str.strip()
    out = out[out["review_text"].ne("")].copy()

    def normalize_label(value):
        if pd.isna(value):
            return None
        s = str(value).strip().lower()
        if s in {"0", "genuine", "truthful", "real", "normal", "not fake", "not spam", "non-spam"}:
            return 0
        if s in {"1", "fake", "deceptive", "spam", "fraudulent", "false"}:
            return 1
        try:
            n = int(float(s))
            return n if n in (0, 1) else None
        except ValueError:
            return None

    out["label"] = out["label"].map(normalize_label)
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    if out.empty:
        raise ValueError("No usable labeled reviews were found. Labels must map to 0 (genuine) or 1 (deceptive).")

    # Detect duplicates case-insensitively and normalize whitespace. Keep the
    # original review text for training, but remove normalized text conflicts so
    # the same review cannot leak across labels or train/test partitions.
    out["_normalized_text"] = (out["review_text"].str.lower()
                                .str.replace(r"\s+", " ", regex=True).str.strip())
    conflicts = out.groupby("_normalized_text")["label"].nunique()
    conflict_texts = set(conflicts[conflicts > 1].index)
    if conflict_texts:
        out = out[~out["_normalized_text"].isin(conflict_texts)]
    out = out.drop_duplicates(subset=["_normalized_text"], keep="first")
    out = out.drop(columns=["_normalized_text"]).reset_index(drop=True)
    if out["label"].nunique() != 2:
        raise ValueError("After cleaning, both classes are required: 0=genuine and 1=deceptive.")
    if len(out) < 20:
        raise ValueError(f"Only {len(out)} usable reviews remain. Use a larger labeled dataset (at least 20 rows; preferably hundreds).")
    return out
