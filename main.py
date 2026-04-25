from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# loads  the data
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target # adding labels(0, 1, or 2) to predict

df["species_name"] = df["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

def main():
    visualize()

def visualize():
    sns.pairplot(df, hue="species_name", vars=iris.feature_names)
    plt.suptitle("Iris Feature Pairplot", y=1.02)
    plt.show()
