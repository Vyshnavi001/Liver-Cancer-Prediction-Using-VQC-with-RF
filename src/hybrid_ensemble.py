import numpy as np


def hybrid_predict(rf, vqc, X_test):
    rf_probs = rf.predict_proba(X_test)[:, 1]
    vqc_probs = vqc.predict(X_test[:, :4])

    final_probs = []

    for r, q in zip(rf_probs, vqc_probs):
        # If RF is confident, trust RF
        if r > 0.6 or r < 0.4:
            final_probs.append(r)
        else:
            # Use quantum-assisted correction
            final_probs.append(0.85*r + 0.15*q)

    final_probs = np.array(final_probs)
    final_pred = (final_probs > 0.5).astype(int)

    return final_pred, final_probs
