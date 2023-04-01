from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from evaluate import evaluate_model
import pandas as pd

# Load the iris dataset
df = pd.read_csv(
    r"C:\Users\Sahlo\dev\python-data-science-studies\supervised_learning_demos\iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

# Create a Naive-Bayes classifier object
nb = GaussianNB()

# ? K-fold validation
y_pred = cross_val_predict(
    nb, X, y, cv=5)  # k-fold cross-validation

# evalute the model
evaluate_model(y.values, y_pred, ["Iris-setosa",
               "Iris-versicolor", "Iris-virginica"])
