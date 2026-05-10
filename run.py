from utils import *

X_train, X_test, y_train, y_test = load_digit_dataset()

predicted, clf, tree_predicted, tree_clf = train_test_model(
    X_train,
    X_test,
    y_train
)

performance_model(
    X_test,
    y_test,
    predicted,
    clf
)

performance_model(
    X_test,
    y_test,
    tree_predicted,
    tree_clf
)

saved_model = load_saved_model()

print(saved_model)