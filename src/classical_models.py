from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def train_rf(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split = 5,
        min_samples_leaf = 2,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf


def train_svm(X_train, y_train):
    svm = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced"
    )
    svm.fit(X_train, y_train)
    return svm
