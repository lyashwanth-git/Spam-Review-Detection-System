import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42

dataset = None
X_train = X_test = y_train = y_test = None

accuracy = []
recall = []
precision = []
algorithm_names = []

MODEL_LABELS = [
    "KNN",
    "Naive Bayes",
    "MLP",
    "SVM",
    "Decision Tree",
    "AdaBoost",
    "Random Forest",
]


def add_result(name, y_true, y_pred):
    accuracy.append(accuracy_score(y_true, y_pred) * 100)
    precision.append(
        precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    )
    recall.append(
        recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    )
    algorithm_names.append(name)

    text.insert(
        tk.END,
        f"{name} Precision : {precision[-1]:.2f}%\n"
        f"{name} Recall    : {recall[-1]:.2f}%\n"
        f"{name} Accuracy  : {accuracy[-1]:.2f}%\n\n",
    )


def upload():
    global dataset

    filename = filedialog.askopenfilename(
        title="Select SPAMBASE Dataset",
        filetypes=[
            ("All files", "*.*"),
            ("Data files", "*.data"),
            ("CSV files", "*.csv")
        ]
    )

    if not filename:
        return

    try:
        dataset = pd.read_csv(filename)
        text.delete("1.0", tk.END)
        text.insert(tk.END, f"{filename} Loaded\n\n")
        text.insert(tk.END, dataset.head().to_string())
    except Exception as exc:
        messagebox.showerror("Error", f"Unable to load dataset:\n{exc}")


def preprocess():
    global X_train, X_test, y_train, y_test

    if dataset is None:
        messagebox.showwarning("Dataset Required", "Please upload the SPAMBASE dataset first.")
        return

    values = dataset.values
    cols = values.shape[1] - 1

    X = values[:, :cols]
    y = values[:, cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    accuracy.clear()
    recall.clear()
    precision.clear()
    algorithm_names.clear()

    text.delete("1.0", tk.END)
    text.insert(tk.END, f"Total records found in dataset : {len(X)}\n")
    text.insert(tk.END, "Splitting dataset into train and test.\n")
    text.insert(tk.END, "Training data : 80%\n")
    text.insert(tk.END, "Testing data  : 20%\n")
    text.insert(tk.END, f"Training records : {len(X_train)}\n")
    text.insert(tk.END, f"Testing records  : {len(X_test)}\n")


def check_ready():
    if X_train is None:
        messagebox.showwarning(
            "Preprocessing Required",
            "Please upload and preprocess the dataset first."
        )
        return False
    return True


def run_knn_nb_mlp():
    if not check_ready():
        return

    text.delete("1.0", tk.END)

    models = [
        ("KNN", KNeighborsClassifier(n_neighbors=2)),
        ("Naive Bayes", BernoulliNB(binarize=0.0)),
        ("MLP", MLPClassifier(random_state=RANDOM_STATE, max_iter=300)),
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        add_result(name, y_test, prediction)


def run_svm_dt_adaboost():
    if not check_ready():
        return

    models = [
        ("SVM", SVC()),
        ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ("AdaBoost", AdaBoostClassifier(random_state=RANDOM_STATE)),
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        add_result(name, y_test, prediction)


def run_random_forest():
    if not check_ready():
        return

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    add_result("Random Forest", y_test, prediction)


def show_graph(values, title, ylabel):
    if not values:
        messagebox.showwarning(
            "Run Algorithms First",
            "Please run the algorithms before viewing comparison graphs."
        )
        return

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(algorithm_names)), algorithm_names, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def accuracy_graph():
    show_graph(accuracy, "Accuracy Comparison", "Accuracy (%)")


def recall_graph():
    show_graph(recall, "Recall Comparison", "Recall (%)")


def precision_graph():
    show_graph(precision, "Precision Comparison", "Precision (%)")


def check_review():
    review = simpledialog.askstring("Input", "Enter Review:")

    if not review:
        text.insert(tk.END, "No input given\n")
        return

    spam_words = ["buy", "free", "offer", "win", "click", "money"]

    if any(word in review.lower() for word in spam_words):
        result = "Review is FAKE (Spam)"
    else:
        result = "Review is GENUINE"

    text.insert(tk.END, f"\nRule-based review check: {result}\n\n")


def close():
    main.destroy()


main = tk.Tk()
main.title("Spam Review Detection Using Machine Learning")
main.geometry("1200x700")
main.config(bg="light coral")

font = ("times", 12, "bold")

title = tk.Label(
    main,
    text="Spam Review Detection Using Machine Learning",
    justify=tk.LEFT,
    bg="lavender blush",
    fg="DarkOrchid1",
    font=("times", 15, "bold"),
    height=3,
)
title.pack(fill=tk.X, padx=10, pady=5)

upload_button = tk.Button(
    main, text="Upload SPAMBASE Dataset", command=upload, font=font
)
upload_button.place(x=10, y=100)

preprocess_button = tk.Button(
    main, text="Preprocess Dataset", command=preprocess, font=font
)
preprocess_button.place(x=250, y=100)

first_button = tk.Button(
    main,
    text="Run KNN, Naive Bayes & MLP",
    command=run_knn_nb_mlp,
    font=font,
)
first_button.place(x=450, y=100)

second_button = tk.Button(
    main,
    text="Run SVM, Decision Tree & AdaBoost",
    command=run_svm_dt_adaboost,
    font=font,
)
second_button.place(x=750, y=100)

rf_button = tk.Button(
    main,
    text="Run Random Forest",
    command=run_random_forest,
    font=font,
)
rf_button.place(x=10, y=155)

acc_button = tk.Button(
    main, text="Accuracy Comparison", command=accuracy_graph, font=font
)
acc_button.place(x=230, y=155)

recall_button = tk.Button(
    main, text="Recall Comparison", command=recall_graph, font=font
)
recall_button.place(x=450, y=155)

precision_button = tk.Button(
    main, text="Precision Comparison", command=precision_graph, font=font
)
precision_button.place(x=650, y=155)

check_button = tk.Button(
    main,
    text="Check Review (Rule-Based Demo)",
    command=check_review,
    font=font,
)
check_button.place(x=880, y=155)

text = tk.Text(main, height=24, width=145, font=font)
text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(80, 10))

main.protocol("WM_DELETE_WINDOW", close)
main.mainloop()
