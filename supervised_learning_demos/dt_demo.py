from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn import datasets  # Load the iris dataset

# load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a decision tree classifier object
dt = DecisionTreeClassifier()

# ? Hold-out validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)  # Split using a holdout 20/80
dt.fit(X_train, y_train)  # Train using the training data

# ? K-fold validation
kfold_acurracies = cross_val_score(dt, X, y, cv=5)  # k-fold cross-validation

# Evaluate the classifier using hold-out validation
holdout_accuracy = dt.score(X_test, y_test)
print("Hold-out validation:")
print(
    f'Accuracy: {holdout_accuracy:.2f} (+/- {kfold_acurracies.std() * 2:.2f})')

# Evaluate the classifier using 5-fold cross-validation
print("K-fold cross-validation:")
print(
    f'Accuracy: {kfold_acurracies.mean():.2f} (+/- {kfold_acurracies.std() * 2:.2f})')
