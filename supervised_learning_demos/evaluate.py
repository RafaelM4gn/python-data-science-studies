import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(y_true, y_pred, classes):

    print('\033[1;36m' + " # ------> Calculate accuracy" + '\033[0m')
    accuracy = accuracy_score(y_true, y_pred)
    print('\033[1;32m' + " => Accuracy: {:.1f}%".format(accuracy*100))

    print('\033[1;36m' + " # ------> Calculate precision" + '\033[0m')
    # Calculate precision for each class
    print(" # Percentage of true positive predictions among all positive predictions made by the model.")
    for label in classes:  # Calculate precision for each class
        precision = precision_score(y_true, y_pred, labels=[
            label], average='macro')
        print(
            " - Precision for class {}: \033[1m{:.1f}%\033[0m".format(label, precision*100))

    # Calculate precision with macro averaging
    precision = precision_score(y_true, y_pred, average='macro')
    print('\033[1;32m' + " => Precision: {:.1f}%".format(precision*100))

    print('\033[1;36m' + " # ------> Calculate recall" + '\033[0m')
    # Calculate recall for each class
    print(" # Percentage of true positive predictions among all actual positive instances in the dataset.")
    for label in classes:
        recall = recall_score(y_true, y_pred, labels=[
            label], average='weighted')
        print(
            " - Recall for class {}: \033[1m{:.1f}%\033[0m".format(label, recall*100))

    # calculate recall with macro averaging
    recall = recall_score(y_true, y_pred, average='macro')
    print('\033[1;32m' + " => Recall: {:.1f}%".format(recall*100))

    print('\033[1;36m' + " # ------> Calculate F1-score" + '\033[0m')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print('\033[1;32m' + " => F1-score: {:.1f}%".format(f1*100))

    print('\033[1;36m' + " # ------> Calculate confusion matrix" + '\033[0m')

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    plt.show()
