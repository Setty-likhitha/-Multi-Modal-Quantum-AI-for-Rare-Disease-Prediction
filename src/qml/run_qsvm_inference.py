import sys

from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
print("Running QSVM inference inside quantum_core...")
import joblib
import pandas as pd

# load model + scaler
qsvm = joblib.load(project_root / "models" / "qsvm.joblib")
scaler = joblib.load(project_root / "models" / "qml_scaler.joblib")

# load input features sent from dashboard
features = pd.read_csv(project_root / "temp_qml_input.csv").values
features_scaled = scaler.transform(features)

# predict probability
probs = qsvm.predict_proba(features_scaled)
score = float(probs[0,1]) if probs.shape[1] > 1 else float(probs[0,0])

# save output
pd.DataFrame({"qsvm_score":[score]}).to_csv(
    project_root / "temp_qsvm_out.csv", index=False
)

print("QSVM inference complete")
