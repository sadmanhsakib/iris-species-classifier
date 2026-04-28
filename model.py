import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

iris = load_iris()

X = iris.data
y = iris.target


def main():
    X_train, X_test, y_train, y_test = split_data()

    lr_f1, lr_train_score, lr_test_score = logistic_classification(X_train, X_test, y_train, y_test)
    rf_f1, rf_train_score, rf_test_score = random_forest_classification(X_train, X_test, y_train, y_test)

    print(f"{lr_f1}, {rf_f1}")
    print(f"{lr_test_score}, {rf_test_score}")
    print(f"{lr_train_score}, {rf_train_score}")

def split_data():
    # spliting the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,  # reserving 20% of the data for testing
        random_state=42,  # seed for reproducibility
        stratify=y,  # preserve class balance in both splits
    )
    return X_train, X_test, y_train, y_test


def logistic_classification(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> tuple[float, float, float]:
    # LogisticRegression model (does classification)
    lr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )

    # training the model with training data
    lr_pipeline.fit(X_train, y_train)

    # model performance on seen data
    train_score = lr_pipeline.score(X_train, y_train)
    # model performance on unseen data
    test_score = lr_pipeline.score(X_test, y_test)
    f1 = f1_score(y_test, lr_pipeline.predict(X_test), average="weighted")

    f1 = round(f1*100, 2)
    train_score = round(train_score*100, 2)
    test_score = round(test_score*100, 2)

    return f1, train_score, test_score


def random_forest_classification(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> tuple[float, float, float]:
    # random forest pipeline
    rf_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # training the model with training data
    rf_pipeline.fit(X_train, y_train)

    # model performance on seen data
    train_score = rf_pipeline.score(X_train, y_train)
    # model performance on unseen data
    test_score = rf_pipeline.score(X_test, y_test)
    f1 = f1_score(y_test, rf_pipeline.predict(X_test), average="weighted")

    f1 = round(f1 * 100, 2)
    train_score = round(train_score * 100, 2)
    test_score = round(test_score * 100, 2)

    return f1, train_score, test_score


main()
