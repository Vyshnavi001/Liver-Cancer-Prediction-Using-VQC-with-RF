import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


def quantum_combined_plots(rf_quantum, X_train_aug, y_train, X_test_aug, y_test):

    train_acc = []
    val_acc = []

    n_trees = len(rf_quantum.estimators_)

    # Simulate epochs using trees
    for i in range(1, n_trees + 1):

        train_pred = np.mean(
            [tree.predict(X_train_aug) for tree in rf_quantum.estimators_[:i]],
            axis=0
        ).round()

        val_pred = np.mean(
            [tree.predict(X_test_aug) for tree in rf_quantum.estimators_[:i]],
            axis=0
        ).round()

        train_acc.append(np.mean(train_pred == y_train))
        val_acc.append(np.mean(val_pred == y_test))


    train_loss = 1 - np.array(train_acc)
    val_loss = 1 - np.array(val_acc)


    # Predictions
    y_pred = rf_quantum.predict(X_test_aug)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC
    y_prob = rf_quantum.predict_proba(X_test_aug)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)


    # =============================
    # COMBINED FIGURE
    # =============================

    plt.figure(figsize=(14,10))


    # 1️⃣ Accuracy
    plt.subplot(2,2,1)

    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Trees")
    plt.ylabel("Accuracy")
    plt.legend()



    # 2️⃣ Loss
    plt.subplot(2,2,2)

    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Trees")
    plt.ylabel("Loss")
    plt.legend()



    # 3️⃣ Confusion Matrix
    plt.subplot(2,2,3)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", ax=plt.gca(), colorbar=False)

    plt.title("Confusion Matrix")



    # 4️⃣ ROC Curve
    plt.subplot(2,2,4)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()


    plt.tight_layout()

    plt.show()