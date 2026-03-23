import argparse
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from qsvm import QuantumSVM



print("✔ Running QSVM inside quantum_core environment")

parser = argparse.ArgumentParser()

parser.add_argument("--train_features", required=True)
parser.add_argument("--train_labels", required=True)
parser.add_argument("--test_features", required=True)

args = parser.parse_args()

# Load data
X_train = pd.read_csv(args.train_features).values
y_train = pd.read_csv(args.train_labels).values.ravel()
X_test = pd.read_csv(args.test_features).values

from collections import Counter
import numpy as np

print("Label distribution:", Counter(y_train))

# Balance dataset (avoid QSVM predicting only 0)
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0][:len(pos_idx)]

balanced_idx = np.concatenate([pos_idx, neg_idx])

X_train = X_train[balanced_idx]
y_train = y_train[balanced_idx]

print("Balanced label distribution:", Counter(y_train))


if X_train.size == 0 or X_test.size == 0:
    raise ValueError("Training or test features are empty")

if len(y_train) == 0:
    raise ValueError("Training labels are empty")


import numpy as np

train_min = X_train.min(axis=0)
train_max = X_train.max(axis=0)

X_train = (X_train - train_min) / (train_max - train_min + 1e-8)
X_test  = (X_test  - train_min) / (train_max - train_min + 1e-8)


print("Training Quantum SVM...")


model = QuantumSVM(
    num_features=X_train.shape[1],
    feature_map_reps=4,
    entanglement="full",
    C=2.5,
    use_quantum=True
)


print("Using quantum kernel:", model.use_quantum)
model.fit(X_train, y_train)
# =============================
# SAVE TRAINED QSVM MODEL
# =============================
import joblib

project_root = Path(__file__).parent.parent.parent
model_path = project_root / "models" / "qsvm.joblib"

# create models folder if missing
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_path)
print("QSVM model saved to:", model_path)

print("Predicting...")

preds = model.predict(X_test)

# Save predictions
project_root = Path(__file__).parent.parent.parent
output_path = project_root / "quantum_svm_predictions.csv"

# if file exists, remove first
if output_path.exists():
    output_path.unlink()

pd.DataFrame(preds).to_csv(output_path, index=False)
print("Predictions saved to:", output_path)


print("✔ QSVM finished")
