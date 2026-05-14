import shap
import numpy as np
import matplotlib.pyplot as plt


def show_shap(rf, X_train, X_test, feature_names):

    explainer = shap.TreeExplainer(rf)

    shap_values = explainer.shap_values(
        X_test,
        check_additivity=False
    )

    # ⭐ HANDLE ALL CASES SAFELY

    if isinstance(shap_values, list):

        shap_vals = shap_values[1]

    elif len(shap_values.shape) == 3:

        shap_vals = shap_values[:, :, 1]

    else:

        shap_vals = shap_values


    # ⭐ CRITICAL FIX
    mean_shap = np.abs(shap_vals).mean(axis=0)

    # convert to 1D if needed
    if len(mean_shap.shape) > 1:

        mean_shap = mean_shap.mean(axis=1)


    # match feature names length
    feature_names = feature_names[:len(mean_shap)]


    plt.figure(figsize=(10,6))

    plt.barh(range(len(mean_shap)), mean_shap)

    plt.yticks(range(len(mean_shap)), feature_names)

    plt.xlabel("Mean |SHAP value|")

    plt.title("Feature Importance using SHAP")

    plt.tight_layout()

    plt.show()