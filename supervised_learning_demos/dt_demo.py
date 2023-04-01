from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from evaluate import evaluate_model
import pandas as pd

# Load the iris dataset and split it into features and class labels
df = pd.read_csv(
    r"C:\Users\Sahlo\dev\python-data-science-studies\supervised_learning_demos\iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

# Create a decision tree classifier object
dt = DecisionTreeClassifier()

# ? K-fold validation
y_pred = cross_val_predict(dt, X, y, cv=5)  # k-fold cross-validation

# evalute the model
evaluate_model(y.values, y_pred, ["Iris-setosa",
               "Iris-versicolor", "Iris-virginica"])
