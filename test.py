import numpy as np
import pandas as pd

from src.data_preprocessing import load_and_preprocess
from src.classical_models import train_rf
from src.quantum_model import train_vqc
from sklearn.ensemble import RandomForestClassifier

# =====================================
# 1. LOAD DATA (same as main.py)
# =====================================
(X_train, X_test, y_train, y_test), feature_names = load_and_preprocess("data/liver_cancer.csv")

# =====================================
# 2. TRAIN MODELS
# =====================================
rf = train_rf(X_train, y_train)
vqc = train_vqc(X_train, y_train)

# =====================================
# 3. CREATE QUANTUM FEATURES (TRAIN)
# =====================================
vqc_train_feat = vqc.predict(X_train[:, :4]).reshape(-1, 1)
X_train_aug = np.hstack([X_train, vqc_train_feat])

# =====================================
# 4. TRAIN QUANTUM-ENHANCED RF
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
# 5. USER INPUT
# =====================================
print("\nEnter patient details:")

age = float(input("Age: "))
tb = float(input("Total Bilirubin: "))
db = float(input("Direct Bilirubin: "))
alk = float(input("Alkaline Phosphotase: "))
alt = float(input("Alamine Aminotransferase: "))
ast = float(input("Aspartate Aminotransferase: "))
tp = float(input("Total Proteins: "))
alb = float(input("Albumin: "))
ratio = float(input("Albumin/Globulin Ratio: "))

# =====================================
# 6. CREATE DATAFRAME (RAW INPUT)
# =====================================
input_df = pd.DataFrame([{
    "Age": age,
    "Total_Bilirubin": tb,
    "Direct_Bilirubin": db,
    "Alkaline_Phosphotase": alk,
    "Alamine_Aminotransferase": alt,
    "Aspartate_Aminotransferase": ast,
    "Total_Protiens": tp,
    "Albumin": alb,
    "Albumin_and_Globulin_Ratio": ratio
}])

# =====================================
# 7. MATCH TRAINING FEATURES
# =====================================
input_data = input_df.reindex(columns=feature_names, fill_value=0)
input_data = input_data.values

# =====================================
# 8. CREATE QUANTUM FEATURE (INPUT)
# =====================================
vqc_feat = vqc.predict(input_data[:, :4]).reshape(-1, 1)

# =====================================
# 9. AUGMENT INPUT
# =====================================
input_aug = np.hstack([input_data, vqc_feat])

# =====================================
# 10. PREDICT
# =====================================
prediction = rf_quantum.predict(input_aug)
prob = rf_quantum.predict_proba(input_aug)

# =====================================
# 11. OUTPUT
# =====================================
print("\n===== RESULT =====")

if prediction[0] == 1:
    print(" The person is likely to have Liver Cancer")
else:
    print(" The person is NOT likely to have Liver Cancer")

print(f"Confidence: {max(prob[0])*100:.2f}%")