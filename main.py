from src.data_preprocessing import load_and_preprocess
from src.classical_models import train_rf, train_svm
from src.quantum_model import train_vqc
from src.hybrid_ensemble import hybrid_predict
from src.explainability import show_shap
from src.quantum_combined_plots import quantum_combined_plots

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# =====================================
# 1. LOAD AND PREPROCESS DATA
# =====================================

(X_train, X_test, y_train, y_test), feature_names = load_and_preprocess("data/liver_cancer.csv")


# =====================================
# 2. TRAIN CLASSICAL MODELS
# =====================================

rf = train_rf(X_train, y_train)

svm = train_svm(X_train, y_train)


# =====================================
# 3. TRAIN QUANTUM MODEL (VQC)
# =====================================

vqc = train_vqc(X_train, y_train)


# =====================================
# 4. EVALUATE INDIVIDUAL MODELS
# =====================================

rf_acc = accuracy_score(y_test, rf.predict(X_test))

svm_acc = accuracy_score(y_test, svm.predict(X_test))

vqc_acc = accuracy_score(y_test, vqc.predict(X_test[:, :4]))


print("\nMODEL ACCURACIES")
print("----------------------------")
print(f"Random Forest Accuracy : {rf_acc*100:.2f}%")
print(f"SVM Accuracy           : {svm_acc*100:.2f}%")
print(f"VQC Accuracy           : {vqc_acc*100:.2f}%")


# =====================================
# 5. CREATE QUANTUM FEATURES
# =====================================

vqc_train_feat = vqc.predict(X_train[:, :4]).reshape(-1, 1)

vqc_test_feat = vqc.predict(X_test[:, :4]).reshape(-1, 1)


# =====================================
# 6. AUGMENT DATASET WITH QUANTUM FEATURE
# =====================================

X_train_aug = np.hstack([X_train, vqc_train_feat])

X_test_aug = np.hstack([X_test, vqc_test_feat])


# =====================================
# 7. TRAIN QUANTUM-ENHANCED RANDOM FOREST
# =====================================

rf_quantum = RandomForestClassifier(
    n_estimators=600,
    max_depth=14,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

rf_quantum.fit(X_train_aug, y_train)


# =====================================
# 8. EVALUATE QUANTUM MODEL
# =====================================

qe_acc = accuracy_score(y_test, rf_quantum.predict(X_test_aug))

print(f"\nQuantum-Enhanced RF Accuracy : {qe_acc*100:.2f}%")


# =====================================
# 9. SHAP EXPLAINABILITY
# =====================================

print("\nGenerating SHAP feature importance...")

show_shap(rf, X_train, X_test, feature_names)


# =====================================
# 10. PERFORMANCE GRAPHS
# =====================================

print("\nGenerating evaluation graphs...")

quantum_combined_plots(
    rf_quantum,
    X_train_aug,
    y_train,
    X_test_aug,
    y_test
)


print("\nExecution Completed Successfully.")