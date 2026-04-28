import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target


def main():
    lr_scores = logistic_classification()
    rf_scores = random_forest_classification()

    print("Logistic Regression:")
    print(f"  Fold scores: {lr_scores.round(4)}")
    print(f"  Mean: {lr_scores.mean():.4f}  Std: {lr_scores.std():.4f}")

    print("\nRandom Forest:")
    print(f"  Fold scores: {rf_scores.round(4)}")
    print(f"  Mean: {rf_scores.mean():.4f}  Std: {rf_scores.std():.4f}")
    
    visualize(lr_scores, rf_scores)


def logistic_classification() -> np.ndarray:
    # LogisticRegression model (does classification)
    lr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )
    
    # cross validating the scores for more reliable score
    lr_scores = cross_val_score(
        lr_pipeline,
        X,
        y,  # Full dataset — cross_val_score handles splitting internally
        cv=5,  # Number of folds
        scoring="accuracy",
    )
    return lr_scores


def random_forest_classification() -> np.ndarray:
    # random forest pipeline
    rf_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    rf_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring="accuracy")
    return rf_scores


def visualize(lr_scores: np.ndarray, rf_scores: np.ndarrays):
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
