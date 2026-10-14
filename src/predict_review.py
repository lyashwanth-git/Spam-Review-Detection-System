"""Prediction helpers. Labels are model predictions, not proof of deception."""
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "review_model.joblib"


def load_model():
    """Load the most recently trained pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Load a labeled dataset and train models first.")
    return joblib.load(MODEL_PATH)


def predict_review(model, review_text):
    """Predict one review and return a label, readable category, and feature clues."""
    if not isinstance(review_text, str) or not review_text.strip():
        raise ValueError("Enter a non-empty review first.")
    text = review_text.strip()
    label = int(predict_labels(model, [text])[0])
    category = "Potentially fake / deceptive" if label == 1 else "Likely genuine"
    result = {
        "label": label,
        "category": category,
        "explanation": influential_terms(model, text),
    }

    # Expose a probability only when the selected estimator supports it. This is a
    # model score, not proof; proper calibration would be needed to interpret it
    # as a real-world probability of deception.
    classifier = model.steps[-1][1] if hasattr(model, "steps") else None
    if classifier is not None and hasattr(classifier, "predict_proba"):
        try:
            probabilities = model.predict_proba([text])[0]
            classes = list(classifier.classes_)
            if label in classes:
                result["model_score"] = float(probabilities[classes.index(label)])
                result["model_score_note"] = "Uncalibrated model score; not the real-world probability that this review is fake."
        except Exception:
            pass
    elif classifier is not None and hasattr(classifier, "decision_function"):
        try:
            result["decision_score"] = float(model.decision_function([text])[0])
        except Exception:
            pass
    return result


def predict_labels(model, texts):
    """Apply the validation-selected threshold when the trained model has one."""
    threshold = getattr(model, "decision_threshold_", None)
    if threshold is not None and hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            probabilities = model.predict_proba(texts)[:, classes.index(1)]
            return (probabilities >= float(threshold)).astype(int)
    return model.predict(texts)


def influential_terms(model, review_text, limit=6):
    """Return linear-model feature contributions for TF-IDF or FeatureUnion pipelines."""
    try:
        if not hasattr(model, "named_steps"):
            return []
        classifier = model.named_steps.get("classifier")
        if classifier is None or not hasattr(classifier, "coef_"):
            return []

        # The selected pipeline may use either a single TF-IDF vectorizer or a
        # FeatureUnion combining word and character n-grams.
        if "tfidf" in model.named_steps:
            vectorizer = model.named_steps["tfidf"]
        elif "features" in model.named_steps:
            vectorizer = model.named_steps["features"]
        else:
            return []

        vector = vectorizer.transform([review_text])
        feature_names = vectorizer.get_feature_names_out()
        indices = vector.indices
        if len(indices) == 0:
            return []
        weights = classifier.coef_[0]
        contributions = [(str(feature_names[i]), float(vector[0, i] * weights[i])) for i in indices]
        predicted = int(predict_labels(model, [review_text])[0])
        contributions.sort(key=lambda item: item[1], reverse=(predicted == 1))
        return [f"{term} ({score:+.3f})" for term, score in contributions[:limit]]
    except Exception:
        # Explanations are a supporting feature; never let them break prediction.
        return []
