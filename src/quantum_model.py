from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA


def train_vqc(X_train, y_train):
    # Use small subset for quantum model
    X_q = X_train[:120, :4]
    y_q = y_train[:120]   # ✅ FIXED

    feature_map = ZZFeatureMap(4)
    ansatz = TwoLocal(4, "ry", "cz")

    optimizer = COBYLA(maxiter=50)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer
    )

    vqc.fit(X_q, y_q)
    return vqc