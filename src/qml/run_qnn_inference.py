import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
import joblib
import pandas as pd
print("Running QNN inference inside quantum_core...")


# load model + scaler
qnn = joblib.load(project_root / "models" / "qnn.joblib")
scaler = joblib.load(project_root / "models" / "qml_scaler.joblib")

# load dashboard features
features = pd.read_csv(project_root / "temp_qml_input.csv").values
features_scaled = scaler.transform(features)

# predict
probs = qnn.predict_proba(features_scaled)
score = float(probs[0,1]) if probs.shape[1] > 1 else float(probs[0,0])

# save
pd.DataFrame({"qnn_score":[score]}).to_csv(
    project_root / "temp_qnn_out.csv", index=False
)

print("QNN inference complete")
