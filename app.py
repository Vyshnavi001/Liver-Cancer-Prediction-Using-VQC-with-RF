from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

from src.data_preprocessing import load_and_preprocess
from src.quantum_model import train_vqc
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# =============================
# LOAD DATA
# =============================
(X_train, X_test, y_train, y_test), feature_names, scaler = load_and_preprocess("data/liver_cancer.csv")

# =============================
# SELECT EXACT 4 FEATURES FOR VQC (BY NAME)
# =============================
# vqc_features = [
#     "Age",
#     "Total_Bilirubin",
#     "Direct_Bilirubin",
#     "Alkaline_Phosphotase"
# ]

# # Get indices safely
# vqc_feature_idx = []
# for f in vqc_features:
#     if f in feature_names:
#         vqc_feature_idx.append(feature_names.get_loc(f))
# =============================
# SELECT FIRST 4 FEATURES (SAFE)
# =============================
vqc_feature_idx = list(range(4))   # always 4 features
# Ensure exactly 4 features
if len(vqc_feature_idx) != 4:
    raise ValueError("VQC requires exactly 4 valid features. Check dataset column names.")

# =============================
# TRAIN VQC
# =============================
vqc = train_vqc(X_train[:, vqc_feature_idx], y_train)

# =============================
# CREATE QUANTUM FEATURE
# =============================
vqc_train_feat = vqc.predict(X_train[:, vqc_feature_idx]).reshape(-1, 1)

# =============================
# AUGMENT DATA
# =============================
X_train_aug = np.hstack([X_train, vqc_train_feat])

# =============================
# TRAIN RF
# =============================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    random_state=42
)

rf.fit(X_train_aug, y_train)

# =============================
# HELPER FUNCTIONS
# =============================

def get_risk(prediction, confidence):
    if prediction == 0:
        return "Low Risk"
    if confidence < 60:
        return "Medium Risk"
    elif confidence < 85:
        return "High Risk"
    else:
        return "Critical Risk"

def get_suggestions(data, prediction):
    if prediction == 0:
        return ["All parameters look normal. Maintain a healthy lifestyle."]

    suggestions = []

    if float(data['tb']) > 1.2:
        suggestions.append("High Bilirubin → Liver test recommended")
    if float(data['alt']) > 56:
        suggestions.append("High ALT → Possible liver damage")
    if float(data['ast']) > 40:
        suggestions.append("High AST → Monitor liver condition")
    if float(data['alb']) < 3.5:
        suggestions.append("Low Albumin → Improve protein intake")

    return suggestions if suggestions else ["Consult doctor"]

# =============================
# ROUTES
# =============================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # =============================
    # CREATE INPUT
    # =============================
    input_df = pd.DataFrame([{
        "Age": float(data['age']),
        "Total_Bilirubin": float(data['tb']),
        "Direct_Bilirubin": float(data['db']),
        "Alkaline_Phosphotase": float(data['alk']),
        "Alamine_Aminotransferase": float(data['alt']),
        "Aspartate_Aminotransferase": float(data['ast']),
        "Total_Protiens": float(data['tp']),
        "Albumin": float(data['alb']),
        "Albumin_and_Globulin_Ratio": float(data['ratio'])
    }])

    # =============================
    # MATCH + SCALE
    # =============================
    input_data = input_df.reindex(columns=feature_names, fill_value=0)
    input_data = scaler.transform(input_data)

    # =============================
    # ADD VQC FEATURE
    # =============================
    vqc_feat = vqc.predict(input_data[:, vqc_feature_idx]).reshape(-1, 1)
    input_aug = np.hstack([input_data, vqc_feat])

    # =============================
    # PREDICT
    # =============================
    prediction = int(rf.predict(input_aug)[0])
    prob = rf.predict_proba(input_aug)[0]

    confidence = round(max(prob) * 100, 2)
    result = "Liver Cancer Detected" if prediction == 1 else "No Liver Cancer"

    # =============================
    # RULE-BASED CORRECTION (VERY IMPORTANT)
    # =============================
    if (
        float(data['tb']) > 2.0 and
        float(data['alt']) > 80 and
        float(data['ast']) > 80 and
        float(data['alb']) < 3.2
    ):
        prediction = 1
        result = "Liver Cancer Detected"
        confidence = max(confidence, 85)

    risk = get_risk(prediction, confidence)
    suggestions = get_suggestions(data, prediction)

    return jsonify({
        "result": result,
        "confidence": confidence,
        "risk": risk,
        "suggestions": suggestions
    })


# =============================
# RUN APP
# =============================
if __name__ == '__main__':
    app.run(debug=True)