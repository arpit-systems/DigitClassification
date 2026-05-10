import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


def load_digit_dataset():

    digits = datasets.load_digits()

    n_samples = len(digits.images)

    data = digits.images.reshape((n_samples, -1))

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        digits.target,
        test_size=0.5,
        shuffle=False
    )

    return X_train, X_test, y_train, y_test


def train_test_model(X_train, X_test, y_train):

    clf = svm.SVC(gamma=0.001)

    tree_clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)

    tree_clf.fit(X_train, y_train)

    dump(clf, "digit_model.joblib")

    predicted = clf.predict(X_test)

    tree_predicted = tree_clf.predict(X_test)

    return predicted, clf, tree_predicted, tree_clf


def load_saved_model():

    model = load("digit_model.joblib")

    return model


def performance_model(X_test, y_test, predicted, clf):

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        y_test,
        predicted
    )

    disp.figure_.suptitle("Confusion Matrix")

    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()