"""Tkinter desktop application for ML-based deceptive review detection."""
import sys
import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_loader import load_reviews
from src.train_models import train_and_evaluate
from src.predict_review import load_model, predict_review
from src.database import save_prediction, get_history

class SpamReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Review Detection — Machine Learning")
        self.root.geometry("1120x760")
        self.root.minsize(900, 650)
        self.root.configure(bg="#f4f6fb")
        self.dataset = None
        self.model = None
        self.metrics = None
        self.model_name = "Not trained"
        self.training_queue = queue.Queue()
        self.training_in_progress = False
        self._build_ui()
        self._log("Welcome! Load the included Amazon product-review dataset or another labeled review CSV.")
        self._log("Expected labels: 0=genuine, 1=fake/deceptive. Predictions are not proof of deception.")

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#242b4a", padx=18, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="Spam Review Detection System", font=("Segoe UI", 20, "bold"),
                 fg="white", bg="#242b4a").pack(anchor="w")
        tk.Label(header, text="TF-IDF • Scikit-learn • Model evaluation • Prediction history",
                 font=("Segoe UI", 10), fg="#d8def7", bg="#242b4a").pack(anchor="w", pady=(3, 0))

        actions = ttk.Frame(self.root, padding=(12, 10))
        actions.pack(fill="x")
        ttk.Button(actions, text="Load Labeled CSV", command=self.load_dataset).pack(side="left", padx=4)
        ttk.Button(actions, text="Load Included Amazon Dataset", command=self.load_bundled_dataset).pack(side="left", padx=4)
        self.train_button = ttk.Button(actions, text="Train & Evaluate", command=self.train_models)
        self.train_button.pack(side="left", padx=4)
        ttk.Button(actions, text="Model Comparison", command=self.show_comparison).pack(side="left", padx=4)
        ttk.Button(actions, text="Confusion Matrix", command=self.show_confusion).pack(side="left", padx=4)
        ttk.Button(actions, text="Batch Predict CSV", command=self.batch_predict).pack(side="left", padx=4)
        ttk.Button(actions, text="Prediction History", command=self.show_history).pack(side="left", padx=4)
        ttk.Button(actions, text="Export History CSV", command=self.export_history).pack(side="left", padx=4)

        body = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = ttk.Frame(body, width=340)
        right.pack(side="right", fill="y")

        ttk.Label(left, text="Review to analyze", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(6, 5))
        self.review_input = tk.Text(left, height=7, wrap="word", font=("Segoe UI", 10), relief="solid", bd=1)
        self.review_input.pack(fill="x")
        ttk.Button(left, text="Predict Review Using Trained Model", command=self.check_review).pack(anchor="w", pady=8)
        self.prediction_label = tk.Label(left, text="Prediction will appear here", anchor="w", justify="left",
                                         font=("Segoe UI", 11, "bold"), bg="#e9edf8", fg="#242b4a", padx=10, pady=10)
        self.prediction_label.pack(fill="x", pady=(0, 8))
        self.explanation = tk.Text(left, height=4, wrap="word", font=("Consolas", 9), state="disabled", relief="flat", bg="#f4f6fb")
        self.explanation.pack(fill="x")

        ttk.Label(left, text="Activity and evaluation log", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(10, 4))
        self.log = tk.Text(left, wrap="word", font=("Consolas", 9), height=16, state="disabled", bg="#101827", fg="#e6edf7", insertbackground="white")
        self.log.pack(fill="both", expand=True)

        ttk.Label(right, text="Dataset summary", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(6, 5))
        self.dataset_summary = tk.Label(right, text="No dataset loaded", justify="left", anchor="nw", bg="white", relief="solid", bd=1, padx=10, pady=10, wraplength=300)
        self.dataset_summary.pack(fill="x")
        ttk.Label(right, text="Selected model results", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(14, 5))
        self.metric_summary = tk.Label(right, text="Train a model to see results", justify="left", anchor="nw", bg="white", relief="solid", bd=1, padx=10, pady=10, wraplength=300)
        self.metric_summary.pack(fill="x")
        note = "Important: predictions indicate text patterns learned from the training corpus. They do not prove that a review is fake."
        ttk.Label(right, text=note, wraplength=300, foreground="#6b5260", justify="left").pack(anchor="w", pady=14)

    def _log(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def load_bundled_dataset(self):
        """Load the cleaned Amazon product-review dataset shipped with the project."""
        path = ROOT / "data" / "amazon_product_reviews_cleaned.csv"
        if not path.exists():
            messagebox.showerror("Dataset missing", f"Could not find the included dataset:\n{path}")
            return
        try:
            df = load_reviews(path)
            self.dataset = df
            counts = df["label"].value_counts().to_dict()
            self.dataset_summary.config(text=(f"File: {path.name}\n\nUsable reviews: {len(df):,}\n"
                f"Genuine (0): {counts.get(0, 0):,}\nDeceptive (1): {counts.get(1, 0):,}\n\n"
                "Source: labeled Amazon product-review corpus."))
            self._log(f"Loaded included Amazon dataset: {len(df):,} unique reviews")
            self._log(f"Genuine: {counts.get(0, 0):,} | Deceptive: {counts.get(1, 0):,}")
            self.model = None
            self.metrics = None
        except Exception as exc:
            messagebox.showerror("Dataset error", str(exc))

    def load_dataset(self):
        path = filedialog.askopenfilename(title="Choose labeled review dataset", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df = load_reviews(path)
            self.dataset = df
            counts = df["label"].value_counts().to_dict()
            self.dataset_summary.config(text=f"File: {Path(path).name}\n\nUsable reviews: {len(df):,}\nGenuine (0): {counts.get(0, 0):,}\nDeceptive (1): {counts.get(1, 0):,}\n\nColumns normalized to review_text and label.")
            self._log(f"Loaded dataset: {path}")
            self._log(f"Usable unique reviews: {len(df)} | Genuine: {counts.get(0, 0)} | Deceptive: {counts.get(1, 0)}")
            self.model = None
        except Exception as exc:
            messagebox.showerror("Dataset error", str(exc))

    def train_models(self):
        """Run model training in a worker thread so Tkinter stays responsive."""
        if self.dataset is None:
            messagebox.showwarning("Dataset required", "Load a labeled review CSV first.")
            return
        if self.training_in_progress:
            return
        self.training_in_progress = True
        self.train_button.configure(state="disabled")
        self.root.config(cursor="watch")
        self._log("\nTraining started in background. The window should remain responsive.")
        self._log("Experiments: Logistic Regression, Multinomial Naive Bayes, and Linear SVM with word TF-IDF.")
        self._log("Model selection uses validation data; the final test split is held out from selection.")
        worker = threading.Thread(target=self._training_worker, args=(self.dataset.copy(),), daemon=True)
        worker.start()
        self.root.after(150, self._check_training_queue)

    def _training_worker(self, dataset):
        try:
            result = train_and_evaluate(dataset)
            self.training_queue.put(("success", result))
        except Exception as exc:
            self.training_queue.put(("error", str(exc)))

    def _check_training_queue(self):
        """Poll the worker result from Tk's main thread; all UI changes stay on this thread."""
        try:
            status, payload = self.training_queue.get_nowait()
        except queue.Empty:
            if self.training_in_progress:
                self.root.after(150, self._check_training_queue)
            return

        self.training_in_progress = False
        self.train_button.configure(state="normal")
        self.root.config(cursor="")

        if status == "error":
            messagebox.showerror("Training error", payload)
            self._log(f"Training failed: {payload}")
            return

        model, metrics = payload
        self.model, self.metrics = model, metrics
        self.model_name = metrics["best_model"]
        self._log("\nTraining and evaluation completed.")
        self._log(f"Training records: {metrics['train_records']} | Test records: {metrics['test_records']}")
        self._log(f"Validation-selected model: {self.model_name} | Validation macro F1: {metrics['best_validation_macro_f1']:.3f}")
        self._log(f"Validation-selected deceptive-class threshold: {metrics.get('decision_threshold', 0.5):.2f}")
        self._log(metrics["classification_report"])
        self.metric_summary.config(text=(f"Best model: {self.model_name}\n\n"
            f"Test accuracy: {metrics['accuracy']:.3f}\n"
            f"Deceptive precision: {metrics['precision']:.3f}\n"
            f"Deceptive recall: {metrics['recall']:.3f}\n"
            f"Deceptive F1: {metrics['f1']:.3f}\n"
            f"Macro F1: {metrics['macro_f1']:.3f}\n"
            f"Decision threshold: {metrics.get('decision_threshold', 0.5):.2f}\n\n"
            f"Validation records: {metrics['validation_records']}\nTest records: {metrics['test_records']}"))
        messagebox.showinfo(
            "Training complete",
            f"Selected model: {self.model_name}\n"
            f"Test F1 (deceptive class): {metrics['f1']:.3f}\n\n"
            "Results saved under outputs/ and model saved under models/."
        )

    def check_review(self):
        review = self.review_input.get("1.0", "end").strip()
        if not review:
            messagebox.showwarning("Review required", "Enter a review first.")
            return
        try:
            if self.model is None:
                self.model = load_model()
                metadata_path = ROOT / "models" / "model_metadata.json"
                if metadata_path.exists():
                    self.model_name = json.loads(metadata_path.read_text(encoding="utf-8")).get("best_model", "Saved model")
            result = predict_review(self.model, review)
            display = f"Prediction: {result['category']}  (label {result['label']})"
            if "model_score" in result:
                display += f"\nModel score for this prediction: {result['model_score']:.1%} (not a calibrated real-world probability)"
            self.prediction_label.config(text=display)
            self.explanation.configure(state="normal")
            self.explanation.delete("1.0", "end")
            self.explanation.insert("end", "Influential terms (model feature contributions, not proof):\n")
            self.explanation.insert("end", "\n".join(result["explanation"]) if result["explanation"] else "Term explanations are unavailable for this model or text.")
            if "model_score_note" in result:
                self.explanation.insert("end", "\n\nNote: " + result["model_score_note"])
            self.explanation.configure(state="disabled")
            save_prediction(review, result["label"], result["category"], self.model_name)
            self._log(f"Prediction: {result['category']} | {review[:110].replace(chr(10), ' ')}")
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))

    def batch_predict(self):
        if self.model is None:
            try:
                self.model = load_model()
                metadata_path = ROOT / "models" / "model_metadata.json"
                if metadata_path.exists():
                    self.model_name = json.loads(metadata_path.read_text(encoding="utf-8")).get("best_model", "Saved model")
            except Exception as exc:
                messagebox.showwarning("Model required", str(exc))
                return
        source = filedialog.askopenfilename(title="Choose CSV of reviews", filetypes=[("CSV files", "*.csv")])
        if not source:
            return
        try:
            batch = pd.read_csv(source, encoding="utf-8-sig")
            aliases = {str(c).strip().lower(): c for c in batch.columns}
            text_col = next((aliases[c] for c in ("review_text", "review", "text", "content", "review_content") if c in aliases), None)
            if text_col is None:
                raise ValueError("CSV needs a review_text, review, text, or content column.")
            records = []
            for value in batch[text_col].fillna("").astype(str):
                review = value.strip()
                if not review:
                    records.append({"review_text": value, "predicted_label": None, "prediction": "Skipped: empty review", "model_score": None, "influential_terms": ""})
                    continue
                result = predict_review(self.model, review)
                records.append({"review_text": review, "predicted_label": result["label"], "prediction": result["category"], "model_score": result.get("model_score"), "influential_terms": "; ".join(result["explanation"])})
                save_prediction(review, result["label"], result["category"], self.model_name)
            out = pd.DataFrame(records)
            destination = filedialog.asksaveasfilename(title="Save batch predictions", defaultextension=".csv", initialfile="review_predictions.csv", filetypes=[("CSV files", "*.csv")])
            if not destination:
                return
            out.to_csv(destination, index=False)
            self._log(f"Batch prediction completed: {len(out)} rows from {Path(source).name}")
            messagebox.showinfo("Batch prediction complete", f"Processed {len(out)} rows.\nSaved results to:\n{destination}")
        except Exception as exc:
            messagebox.showerror("Batch prediction error", str(exc))

    def show_comparison(self):
        path = ROOT / "outputs" / "evaluation_results.csv"
        if not path.exists():
            messagebox.showwarning("No results", "Train and evaluate models first.")
            return
        df = pd.read_csv(path)
        metric_col = "Validation Macro F1" if "Validation Macro F1" in df.columns else "CV Macro F1"
        ax = df.plot(x="Model", y=metric_col, kind="bar", legend=False, figsize=(8, 4), ylim=(0, 1), title="Model Selection Comparison")
        ax.set_ylabel("Validation Macro F1")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.show()

    def show_confusion(self):
        """Display the test confusion matrix inside a Tk window (no separate plt.show window)."""
        try:
            if self.metrics is None:
                messagebox.showwarning(
                    "No results",
                    "No evaluation results are available in this app session. Click Train & Evaluate, wait for it to finish, and try again."
                )
                self._log("Confusion Matrix: unavailable because this session has no evaluation metrics.")
                return

            cm = self.metrics.get("confusion_matrix")
            if cm is None:
                messagebox.showerror("Missing results", "The saved evaluation metrics do not contain a confusion matrix. Please train and evaluate again.")
                return

            win = tk.Toplevel(self.root)
            win.title(f"Confusion Matrix — {self.model_name}")
            win.geometry("720x600")
            win.transient(self.root)
            fig = Figure(figsize=(6.5, 5.0), dpi=100)
            ax = fig.add_subplot(111)
            cm = np.asarray(cm, dtype=int)  # metrics JSON stores this as a list; Matplotlib expects an array
            if cm.shape != (2, 2):
                raise ValueError(f"Expected a 2x2 confusion matrix, got shape {cm.shape}")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Genuine", "Deceptive"])
            disp.plot(ax=ax, values_format="d", colorbar=False)
            ax.set_title(f"Held-out Test Confusion Matrix\n{self.model_name}")
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            ttk.Label(win, text="Rows = actual labels; columns = predicted labels.").pack(pady=(0, 10))
            ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 10))
            self._log("Opened held-out test confusion matrix.")
        except Exception as exc:
            self._log(f"Confusion Matrix error: {exc}")
            messagebox.showerror("Confusion Matrix error", f"Could not display the confusion matrix:\n{exc}")

    def show_history(self):
        rows = get_history()
        win = tk.Toplevel(self.root)
        win.title("Prediction History")
        win.geometry("950x450")
        cols = ("id", "created_at", "predicted_category", "model_name", "review_text")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        labels = {"id":"ID", "created_at":"Date/time", "predicted_category":"Prediction", "model_name":"Model", "review_text":"Review"}
        widths = {"id":50, "created_at":160, "predicted_category":190, "model_name":150, "review_text":390}
        for col in cols:
            tree.heading(col, text=labels[col]); tree.column(col, width=widths[col], anchor="w")
        for row in rows:
            tree.insert("", "end", values=(row[0], row[1], row[4], row[5], row[2]))
        tree.pack(fill="both", expand=True)
        self._log(f"Opened prediction history ({len(rows)} records).")

    def export_history(self):
        rows = get_history()
        if not rows:
            messagebox.showinfo("No history", "There are no predictions to export yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile="review_prediction_history.csv")
        if not path:
            return
        df = pd.DataFrame(rows, columns=["id", "created_at", "review_text", "predicted_label", "predicted_category", "model_name"])
        df.to_csv(path, index=False)
        messagebox.showinfo("Export complete", f"Exported {len(df)} records to:\n{path}")


def main():
    root = tk.Tk()
    SpamReviewApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
