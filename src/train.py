"""
Training script for HGPS Prediction Models.

This script trains all models (Classical ML, CNN, QML) and saves them
for production deployment.

Usage:
    python -m src.train --all
    python -m src.train --classical
    python -m src.train --qml
    python -m src.train --cnn
"""
from sklearn.model_selection import StratifiedKFold

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data import (
    TabularPreprocessor,
    create_data_splits,
    generate_hgps_tabular_data,
    get_qml_features,
)
from src.models import ClassicalTabularModels


# Ensure directories exist
settings.paths.ensure_dirs()

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.paths.logs_dir / "training.log"),
    ],
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates training of all model types."""

    def __init__(self):
        self.settings = settings
        self.settings.paths.ensure_dirs()
        self.results = {}
        self.device = self._get_device()

    def _get_device(self) -> str:
        if self.settings.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.settings.device

    def generate_training_data(self) -> tuple:
        """Generate and split training data."""
        logger.info("Generating synthetic training data...")

        df = generate_hgps_tabular_data(
            n_hgps=self.settings.model.n_hgps_train,
            n_controls=self.settings.model.n_controls_train,
            random_state=self.settings.model.random_state,
        )

        train_df, val_df, test_df = create_data_splits(
            df,
            test_size=self.settings.model.test_size,
            val_size=self.settings.model.val_size,
            random_state=self.settings.model.random_state,
        )

        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Save data splits
        train_df.to_csv(self.settings.paths.data_dir / "train.csv", index=False)
        val_df.to_csv(self.settings.paths.data_dir / "val.csv", index=False)
        test_df.to_csv(self.settings.paths.data_dir / "test.csv", index=False)

        return train_df, val_df, test_df

    def train_preprocessor(self, train_df: pd.DataFrame) -> TabularPreprocessor:
        """Fit and save the preprocessor."""
        logger.info("Fitting preprocessor...")

        preprocessor = TabularPreprocessor()
        preprocessor.fit(train_df)

        # Save preprocessor
        preprocessor_path = self.settings.paths.models_dir / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        return preprocessor
    def train_classical_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        preprocessor: TabularPreprocessor,
    ) -> dict:

        logger.info("=" * 50)
        logger.info("Training Classical ML Models with Stratified 5-Fold CV")
        logger.info("=" * 50)

        # Combine train + val for cross-validation
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)

        y = full_train_df["risk_label"].values

        skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.settings.model.random_state
        )

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, y), 1):
            logger.info(f"Fold {fold}/5")
            fold_train_df = full_train_df.iloc[train_idx]
            fold_val_df = full_train_df.iloc[val_idx]

            fold_preprocessor = TabularPreprocessor()
            fold_preprocessor.fit(fold_train_df)

            X_train_fold = fold_preprocessor.transform(fold_train_df)
            X_val_fold = fold_preprocessor.transform(fold_val_df)

            y_train_fold = fold_train_df["risk_label"].values
            y_val_fold = fold_val_df["risk_label"].values


            classical_models = ClassicalTabularModels(
                calibrate=True,
                random_state=self.settings.model.random_state
            )

            classical_models.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

            fold_metrics = classical_models.evaluate(X_val_fold, y_val_fold)
            fold_results.append(fold_metrics)

        # Average results across folds
        averaged_results = {}

        for model_name in fold_results[0].keys():
            averaged_results[model_name] = {
                "accuracy_mean": np.mean([f[model_name]["accuracy"] for f in fold_results]),
                "accuracy_std": np.std([f[model_name]["accuracy"] for f in fold_results]),
                "f1_mean": np.mean([f[model_name]["f1"] for f in fold_results]),
                "f1_std": np.std([f[model_name]["f1"] for f in fold_results]),
            }

            logger.info(
                f"{model_name}: "
                f"F1={averaged_results[model_name]['f1_mean']:.4f} ± "
                f"{averaged_results[model_name]['f1_std']:.4f}"
            )
        final_preprocessor = TabularPreprocessor()
        final_preprocessor.fit(full_train_df)

        X_full = final_preprocessor.transform(full_train_df)
        # Retrain final model on full data
        final_models = ClassicalTabularModels(
            calibrate=True,
            random_state=self.settings.model.random_state
        )

        final_models.fit(X_full, y, X_full, y)

        joblib.dump(final_models, self.settings.paths.models_dir / "classical_models.joblib")
        joblib.dump(final_preprocessor, self.settings.paths.models_dir / "preprocessor.joblib")

        logger.info(f"Final classical models saved")

        self.results["classical"] = averaged_results
        return averaged_results


    def train_qml_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        preprocessor: TabularPreprocessor,
    ) -> dict:
        """Train and evaluate Quantum ML models."""
        logger.info("=" * 50)
        logger.info("Training Quantum ML Models")
        logger.info("=" * 50)

        try:
            

            # Get QML features (uses default 6 features for quantum circuits)
            X_train, y_train, _ = get_qml_features(train_df, preprocessor)
            X_val, y_val, _ = get_qml_features(val_df, preprocessor)
            X_test, y_test, _ = get_qml_features(test_df, preprocessor)
            X_train = X_train[:120]
            y_train = y_train[:120]
            X_test  = X_test[:60]
            y_test  = y_test[:60]
            
            # Normalize features for quantum encoding
            # Normalize features for quantum encoding using scaler
            from sklearn.preprocessing import MinMaxScaler

            qml_scaler = MinMaxScaler()

            X_train = qml_scaler.fit_transform(X_train)
            X_test  = qml_scaler.transform(X_test)

            # Save scaler
            scaler_path = self.settings.paths.models_dir / "qml_scaler.joblib"
            joblib.dump(qml_scaler, scaler_path)
            logger.info(f"QML scaler saved to {scaler_path}")   

            results = {}

            # Train QSVM
            # =============================
            # Run QSVM in quantum_core env
            # =============================
            logger.info("Running Quantum SVM in quantum_core environment...")

            start_time = time.time()

            project_root = Path(__file__).parent.parent

            # Save temp features
            pd.DataFrame(X_train).to_csv(project_root / "temp_qsvm_train.csv", index=False)
            pd.DataFrame(y_train).to_csv(project_root / "temp_qsvm_labels.csv", index=False)
            pd.DataFrame(X_test).to_csv(project_root / "temp_qsvm_test.csv", index=False)
            logger.info(f"QSVM training size: {X_train.shape}, Test size: {X_test.shape}")

            subprocess.run([
                r"C:\Users\my pc\anaconda3\condabin\conda.bat",
                "run", "-n", "quantum_core",
                "python", str(project_root / "src/qml/run_qsvm.py"),
                "--train_features", str(project_root / "temp_qsvm_train.csv"),
                "--train_labels", str(project_root / "temp_qsvm_labels.csv"),
                "--test_features", str(project_root / "temp_qsvm_test.csv")
            ], check=True)

            # Load predictions
            y_pred_qsvm = pd.read_csv(project_root / "quantum_svm_predictions.csv").values.ravel()

            qsvm_time = time.time() - start_time
            logger.info(f"Quantum QSVM executed in {qsvm_time:.2f}s")

            results["QSVM"] = {
                "accuracy": accuracy_score(y_test, y_pred_qsvm),
                "precision": precision_score(y_test, y_pred_qsvm, zero_division=0),
                "recall": recall_score(y_test, y_pred_qsvm, zero_division=0),
                "f1": f1_score(y_test, y_pred_qsvm, zero_division=0),
            }


            #Train QNN
            # REAL QNN via quantum_core environment
            logger.info("Running Quantum Neural Network in quantum_core environment...")
            start_time = time.time()

            # Save features to temporary files
            project_root = Path(__file__).parent.parent
            temp_dir = project_root

            pd.DataFrame(X_train).to_csv(temp_dir / "temp_q_train.csv", index=False)
            pd.DataFrame(y_train).to_csv(temp_dir / "temp_q_labels.csv", index=False)
            pd.DataFrame(X_test).to_csv(temp_dir / "temp_q_test.csv", index=False)


            # Call quantum_core env to execute QNN
            
            subprocess.run([
                r"C:\Users\my pc\anaconda3\condabin\conda.bat",
                "run", "-n", "quantum_core",
                "python", str(project_root / "src/qml/qnn.py"),
                "--train_features", str(project_root / "temp_q_train.csv"),
                "--train_labels", str(project_root / "temp_q_labels.csv"),
                "--test_features", str(project_root / "temp_q_test.csv")
            ], check=True)

            # Load predictions from quantum execution
            y_pred_qnn = pd.read_csv(project_root / "quantum_predictions.csv").values.ravel()

            qnn_time = time.time() - start_time
            logger.info(f"Quantum QNN executed in {qnn_time:.2f}s")

            results["QNN"] = {
                "accuracy": accuracy_score(y_test, y_pred_qnn),
                "precision": precision_score(y_test, y_pred_qnn, zero_division=0),
                "recall": recall_score(y_test, y_pred_qnn, zero_division=0),
                "f1": f1_score(y_test, y_pred_qnn, zero_division=0),
            }

            logger.info(f"Quantum QNN: Accuracy={results['QNN']['accuracy']:.4f}, F1={results['QNN']['f1']:.4f}")


            
    

            self.results["qml"] = results
            return results

        except ImportError as e:
            logger.warning(f"QML dependencies not available: {e}")
            return {}

    def train_cnn_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        """Train and evaluate CNN model for face analysis."""
        logger.info("=" * 50)
        logger.info("Training CNN Face Model")
        logger.info("=" * 50)

        try:
            from src.features.face_cnn import FaceCNN

            # Create CNN model
            cnn = FaceCNN(
                pretrained=self.settings.model.cnn_pretrained,
                embedding_dim=256,
                num_risk_classes=2,
                num_progression_classes=3,
            )

            # For now, we save the pre-trained backbone
            # In production, this would be fine-tuned on real face data
            cnn_path = self.settings.paths.models_dir / "face_cnn.pt"
            torch.save(cnn.state_dict(), cnn_path)
            logger.info(f"CNN model saved to {cnn_path}")

            results = {
                "CNN": {
                    "status": "pretrained_backbone_saved",
                    "backbone": "resnet18",
                    "note": "Fine-tune on real facial data for production",
                }
            }

            self.results["cnn"] = results
            return results

        except Exception as e:
            logger.error(f"CNN training failed: {e}")
            return {}

    def save_training_report(self):
        """Save comprehensive training report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "n_hgps": self.settings.model.n_hgps_train,
                "n_controls": self.settings.model.n_controls_train,
                "random_state": self.settings.model.random_state,
                "device": self.device,
            },
            "results": self.results,
        }

        report_path = self.settings.paths.results_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Training report saved to {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        if "classical" in self.results:
            print("\nClassical ML Models (5-Fold CV):")
            for name, metrics in self.results["classical"].items():
                print(
                    f"  {name}: "
                    f"F1={metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}, "
                    f"Accuracy={metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}"
            )


        if "qml" in self.results:
            print("\nQuantum ML Models:")
            for name, metrics in self.results["qml"].items():
                print(f"  {name}: F1={metrics['f1']:.4f}")

        print("\n" + "=" * 60)

    def train_all(self):
        """Train all model types."""
        logger.info("Starting full training pipeline...")

        # Generate data
        train_df, val_df, test_df = self.generate_training_data()

        # Train preprocessor
        preprocessor = self.train_preprocessor(train_df)

        # Train all model types
        self.train_classical_models(train_df, val_df, test_df, preprocessor)
        self.train_qml_models(train_df, val_df, test_df, preprocessor)
        self.train_cnn_model(train_df, val_df, test_df)

        # Save report
        self.save_training_report()

        logger.info("Training pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train HGPS Prediction Models")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--classical", action="store_true", help="Train classical ML models only")
    parser.add_argument("--qml", action="store_true", help="Train QML models only")
    parser.add_argument("--cnn", action="store_true", help="Train CNN model only")

    args = parser.parse_args()

    trainer = ModelTrainer()

    if args.all or not any([args.classical, args.qml, args.cnn]):
        trainer.train_all()
    else:
        # Generate data first
        train_df, val_df, test_df = trainer.generate_training_data()
        preprocessor = trainer.train_preprocessor(train_df)

        if args.classical:
            trainer.train_classical_models(train_df, val_df, test_df, preprocessor)

        if args.qml:
            trainer.train_qml_models(train_df, val_df, test_df, preprocessor)

        if args.cnn:
            trainer.train_cnn_model(train_df, val_df, test_df)

        trainer.save_training_report()


if __name__ == "__main__":
    main()
