import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target


def main():
    X_train, X_test, y_train, y_test = split_data()

    lr_pipeline = logistic_classification()
    rf_pipeline = random_forest_classification()

    # train both pipelines on X_train
    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    # predictions on unseen data
    lr_preds = lr_pipeline.predict(X_test)
    rf_preds = rf_pipeline.predict(X_test)

    """ classification report compares the ground truth labels against the predicted labels.
        Breaks down the performance by class so that we can measure the imbalance.
    """ 
    print("Logistic Regression:")
    print(classification_report(y_test, lr_preds, target_names=iris.target_names))

    print("Random Forest:")
    print(classification_report(y_test, rf_preds, target_names=iris.target_names))

    rf_model = rf_pipeline.named_steps["model"]

    importances = rf_model.feature_importances_
    feature_names = iris.feature_names

    # Sort by importance descending
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(7, 4))
    plt.bar(range(4), importances[indices], color="forestgreen")
    plt.xticks(range(4), [feature_names[i] for i in indices], rotation=15)
    plt.ylabel("Importance Score")
    plt.title("Random Forest — Feature Importances")
    plt.tight_layout()
    plt.show()

def split_data() -> tuple(np.ndarray):
    # spliting the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,  # reserving 20% of the data for testing
        random_state=42,  # seed for reproducibility
        stratify=y,  # preserve class balance in both splits
    )
    return X_train, X_test, y_train, y_test


def logistic_classification() -> sklearn.pipeline.pipeline:
    # LogisticRegression model (does classification)
    lr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )

    return lr_pipeline


def random_forest_classification() -> sklearn.pipeline.pipeline:
    # random forest pipeline
    rf_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    return rf_pipeline


def get_cross_val_score(pipeline: sklearn.pipeline.pipeline) -> np.ndarray:
    # cross validating the scores for more reliable score
    scores = cross_val_score(
        pipeline,
        X,
        y,  # Full dataset — cross_val_score handles splitting internally
        cv=5,  # Number of folds
        scoring="accuracy",
    )
    return scores


def visualize_cross_val_score(lr_scores: np.ndarray, rf_scores: np.ndarrays):
    labels = ["Logistic Regression", "Random Forest"]
    means = [lr_scores.mean(), rf_scores.mean()]
    stds = [lr_scores.std(), rf_scores.std()]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, means, yerr=stds, capsize=8, color=["steelblue", "forestgreen"])
    plt.ylabel("Cross-validated Accuracy")
    plt.title("Model Comparison — 5-Fold CV")
    plt.ylim(0.9, 1.0)

    for bar, mean in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.002,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()

main()
