from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets  # Load the iris dataset

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a kNN classifier object
knn = KNeighborsClassifier(n_neighbors=5)

# ? Hold-out validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)  # Split using a holdout 20/80
knn.fit(X_train, y_train)  # Train  using the training data

# ? K-fold validation
kfold_acurracies = cross_val_score(knn, X, y, cv=5)  # k-fold cross-validation

# Evaluate the classifier using hold-out validation
holdout_accuracy = knn.score(X_test, y_test)
print("Hold-out validation:")
print(
    f'Accuracy: {holdout_accuracy:.2f} (+/- {kfold_acurracies.std() * 2:.2f})')

# Evaluate the classifier using 5-fold cross-validation
print("K-fold cross-validation:")
print(
    f'Accuracy: {kfold_acurracies.mean():.2f} (+/- {kfold_acurracies.std() * 2:.2f})')
