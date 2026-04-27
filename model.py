from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # reserving 20% of the data for testing
    random_state=42, # seed for reproducibility
    stratify=y # preserve class balance in both splits
)

pipeline = Pipeline(
    steps=[("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200))]
)

pipeline.fit(X_train, y_train)

train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

sample = [[5.1, 3.5, 1.4, 0.2]]  # A single new observation

prediction = pipeline.predict(sample)
predicted_species = iris.target_names[prediction[0]]

print(f"Predicted species: {predicted_species}")  # setosa
