"""Production-grade training script for heart disease detection model.

This script handles the complete training pipeline including data loading,
preprocessing, cross-validation, model training, and evaluation.
"""

import os
import logging
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# Add project root to sys.path for imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.signal_cleaner import PPGProcessor
from src.features.hrv_metrics import HRVMetrics
from src.models.classifier import HybridHeartDiseaseClassifier, ModelTrainer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataGenerator:
    """Handle data loading and preprocessing for training."""

    def __init__(
        self,
        sample_rate: float = 100.0,
        ppg_length: int = 1000,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize the data generator.

        Args:
            sample_rate: Sampling rate of PPG signal in Hz.
            ppg_length: Length of PPG signal samples. Default is 1000.
            batch_size: Batch size for training. Default is 32.
        """
        self.sample_rate = sample_rate
        self.ppg_length = ppg_length
        self.batch_size = batch_size
        self.ppg_processor = PPGProcessor(sample_rate=sample_rate)
        self.hrv_extractor = HRVMetrics(sample_rate=sample_rate)

    def load_dataset(
        self, ppg_signals: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess dataset.

        Args:
            ppg_signals: Raw PPG signals of shape (num_samples, signal_length).
            labels: Binary labels (0: healthy, 1: disease) of shape (num_samples,).

        Returns:
            Tuple of (ppg_signals, hrv_features).

        Raises:
            ValueError: If shapes don't match.
        """
        if ppg_signals.shape[0] != labels.shape[0]:
            raise ValueError("Number of signals must match number of labels")

        logger.info(f"Loading dataset with {ppg_signals.shape[0]} samples")

        # Process PPG signals
        cleaned_signals = []
        for idx, signal in enumerate(ppg_signals):
            try:
                cleaned, _ = self.ppg_processor.process_signal(
                    signal, artifact_method="moving_average"
                )
                cleaned_signals.append(cleaned)
            except Exception as e:
                logger.warning(f"Error processing signal {idx}: {e}")
                cleaned_signals.append(signal)

        cleaned_signals = np.array(cleaned_signals)

        # Extract HRV features
        hrv_features = []
        for idx, signal in enumerate(cleaned_signals):
            try:
                metrics = self.hrv_extractor.extract_all_hrv_features(signal)
                feature_vector = np.array(
                    [
                        metrics.get("SDNN_ms", 0),
                        metrics.get("RMSSD_ms", 0),
                        metrics.get("mean_HR_bpm", 0),
                        metrics.get("LF", 0),
                        metrics.get("HF", 0),
                        metrics.get("LF_HF_ratio", 0),
                        metrics.get("LF_norm", 0),
                        metrics.get("HF_norm", 0),
                    ]
                )
                hrv_features.append(feature_vector)
            except Exception as e:
                logger.warning(f"Error extracting HRV features for signal {idx}: {e}")
                hrv_features.append(np.zeros(8))

        hrv_features = np.array(hrv_features)

        # Normalize PPG signals and HRV features
        cleaned_signals = self._normalize_signals(cleaned_signals)
        hrv_features = self._normalize_features(hrv_features)

        logger.info(f"Processed signals shape: {cleaned_signals.shape}")
        logger.info(f"Extracted HRV features shape: {hrv_features.shape}")

        return cleaned_signals, hrv_features

    @staticmethod
    def _normalize_signals(signals: np.ndarray) -> np.ndarray:
        """Normalize PPG signals to [-1, 1] range."""
        normalized = signals.copy().astype(np.float32)
        for idx in range(normalized.shape[0]):
            sig_min = np.min(normalized[idx])
            sig_max = np.max(normalized[idx])
            if sig_max - sig_min > 0:
                normalized[idx] = 2 * (normalized[idx] - sig_min) / (sig_max - sig_min) - 1
        return normalized

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        """Normalize HRV features using z-score normalization."""
        normalized = features.astype(np.float32)
        for col in range(normalized.shape[1]):
            mean = np.mean(normalized[:, col])
            std = np.std(normalized[:, col])
            if std > 0:
                normalized[:, col] = (normalized[:, col] - mean) / std
        return normalized

    def create_tf_dataset(
        self,
        ppg_signals: np.ndarray,
        hrv_features: np.ndarray,
        labels: np.ndarray,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """
        Create TensorFlow Dataset for efficient data loading.

        Args:
            ppg_signals: Cleaned PPG signals.
            hrv_features: Extracted HRV features.
            labels: Binary labels.
            shuffle: Whether to shuffle the dataset. Default is True.

        Returns:
            TensorFlow Dataset object.
        """
        # Expand dimensions for single channel
        ppg_signals = np.expand_dims(ppg_signals, axis=-1)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            ((ppg_signals, hrv_features), labels)
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(labels))

        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset


class TrainingPipeline:
    """Production training pipeline with cross-validation and evaluation."""

    def __init__(
        self,
        model_config: Optional[Dict] = None,
        n_splits: int = 5,
        random_state: int = 42,
        enable_tensorboard: bool = True,
    ) -> None:
        """
        Initialize the training pipeline.

        Args:
            model_config: Model configuration dictionary.
            n_splits: Number of K-Fold splits. Default is 5.
            random_state: Random seed for reproducibility. Default is 42.
            enable_tensorboard: Whether to enable TensorBoard logging.
                                Default is True.
        """
        self.model_config = model_config or {}
        self.n_splits = n_splits
        self.random_state = random_state
        self.enable_tensorboard = enable_tensorboard
        self.cv_results = []
        self.data_generator = DataGenerator()

        # Create output directories
        self.log_dir = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path("checkpoints") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training logs will be saved to {self.log_dir}")
        logger.info(f"Model checkpoints will be saved to {self.checkpoint_dir}")

    def train(
        self,
        ppg_signals: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        validation_split: float = 0.2,
    ) -> Dict:
        """
        Execute stratified K-Fold cross-validation training.

        Args:
            ppg_signals: Array of PPG signals.
            labels: Array of binary labels.
            epochs: Number of training epochs per fold. Default is 100.
            validation_split: Fraction of training data for validation. Default is 0.2.

        Returns:
            Dictionary with cross-validation results.
        """
        logger.info(f"Starting {self.n_splits}-Fold Cross-Validation Training")

        # Load and preprocess dataset
        cleaned_signals, hrv_features = self.data_generator.load_dataset(
            ppg_signals, labels
        )

        # Initialize stratified K-Fold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        fold_idx = 0
        for train_idx, val_idx in skf.split(cleaned_signals, labels):
            fold_idx += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Fold {fold_idx}/{self.n_splits}")
            logger.info(f"{'='*60}")

            # Split data
            X_train_ppg, X_val_ppg = cleaned_signals[train_idx], cleaned_signals[val_idx]
            X_train_hrv, X_val_hrv = hrv_features[train_idx], hrv_features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Create TF datasets
            train_dataset = self.data_generator.create_tf_dataset(
                X_train_ppg, X_train_hrv, y_train, shuffle=True
            )
            val_dataset = self.data_generator.create_tf_dataset(
                X_val_ppg, X_val_hrv, y_val, shuffle=False
            )

            # Build and compile model
            classifier = HybridHeartDiseaseClassifier(**self.model_config)
            model = classifier.build_model()
            classifier.compile_model()

            # Setup callbacks
            callbacks = self._setup_callbacks(fold_idx)

            # Train model
            logger.info(f"Training on {len(y_train)} samples, validating on {len(y_val)} samples")
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
            )

            # Evaluate on validation set
            val_results = self._evaluate_fold(
                model, X_val_ppg, X_val_hrv, y_val, fold_idx
            )
            self.cv_results.append(val_results)

            logger.info(f"Fold {fold_idx} Validation AUC-ROC: {val_results['auc_roc']:.4f}")

        # Summary results
        self._print_cv_summary()

        return self._aggregate_cv_results()

    def _setup_callbacks(self, fold_idx: int) -> List:
        """
        Setup training callbacks.

        Args:
            fold_idx: Current fold index.

        Returns:
            List of Keras callbacks.
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(
                    self.checkpoint_dir / f"model_fold_{fold_idx}_best.h5"
                ),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                verbose=0,
            ),
        ]

        if self.enable_tensorboard and self._is_tensorboard_supported():
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=str(self.log_dir / f"fold_{fold_idx}"),
                    histogram_freq=1,
                    write_graph=True,
                    update_freq="epoch",
                )
            )

        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
            )
        )
        return callbacks

    def _is_tensorboard_supported(self) -> bool:
        """Check whether TensorBoard summaries can be written in this environment."""
        try:
            test_dir = self.log_dir / "tb_test"
            test_dir.mkdir(parents=True, exist_ok=True)
            with tf.summary.create_file_writer(str(test_dir)).as_default():
                tf.summary.scalar("_tb_test_metric", 0.0, step=0)
            return True
        except Exception as exc:
            logger.warning(
                "TensorBoard summary support unavailable: %s. "
                "Skipping TensorBoard callback.",
                exc,
            )
            return False

    def _evaluate_fold(
        self,
        model: keras.Model,
        X_ppg: np.ndarray,
        X_hrv: np.ndarray,
        y_true: np.ndarray,
        fold_idx: int,
    ) -> Dict:
        """
        Evaluate model on a fold.

        Args:
            model: Trained Keras model.
            X_ppg: PPG signals.
            X_hrv: HRV features.
            y_true: True labels.
            fold_idx: Current fold index.

        Returns:
            Dictionary with evaluation metrics.
        """
        # Make predictions
        y_pred_proba = model.predict([X_ppg, X_hrv], verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        cm = confusion_matrix(y_true, y_pred)
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

        logger.info("\nClassification Report:")
        logger.info(
            classification_report(
                y_true, y_pred, target_names=["Healthy", "Disease"]
            )
        )
        logger.info(f"AUC-ROC Score: {auc_roc:.4f}")
        logger.info(f"Sensitivity: {sensitivity:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")

        # Save ROC curve
        self._plot_roc_curve(y_true, y_pred_proba, fold_idx)

        return {
            "fold": fold_idx,
            "auc_roc": auc_roc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

    def _plot_roc_curve(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, fold_idx: int
    ) -> None:
        """
        Plot and save ROC curve.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.
            fold_idx: Current fold index.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - Fold {fold_idx}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.log_dir / f"roc_curve_fold_{fold_idx}.png", dpi=300)
        plt.close()

    def _print_cv_summary(self) -> None:
        """Print cross-validation summary."""
        if not self.cv_results:
            logger.warning("No results to summarize")
            return

        logger.info(f"\n{'='*60}")
        logger.info("Cross-Validation Summary")
        logger.info(f"{'='*60}")
        for result in self.cv_results:
            logger.info(
                f"Fold {result['fold']}: AUC-ROC={result['auc_roc']:.4f}, "
                f"Sensitivity={result['sensitivity']:.4f}, "
                f"Specificity={result['specificity']:.4f}"
            )

    def _aggregate_cv_results(self) -> Dict:
        """
        Aggregate cross-validation results.

        Returns:
            Dictionary with aggregated metrics.
        """
        if not self.cv_results:
            return {}

        auc_scores = [r["auc_roc"] for r in self.cv_results]
        sensitivities = [r["sensitivity"] for r in self.cv_results]
        specificities = [r["specificity"] for r in self.cv_results]

        summary = {
            "mean_auc_roc": np.mean(auc_scores),
            "std_auc_roc": np.std(auc_scores),
            "mean_sensitivity": np.mean(sensitivities),
            "std_sensitivity": np.std(sensitivities),
            "mean_specificity": np.mean(specificities),
            "std_specificity": np.std(specificities),
        }

        logger.info(f"\n{'='*60}")
        logger.info("Aggregated Cross-Validation Results")
        logger.info(f"{'='*60}")
        logger.info(f"Mean AUC-ROC: {summary['mean_auc_roc']:.4f} ± {summary['std_auc_roc']:.4f}")
        logger.info(f"Mean Sensitivity: {summary['mean_sensitivity']:.4f} ± {summary['std_sensitivity']:.4f}")
        logger.info(f"Mean Specificity: {summary['mean_specificity']:.4f} ± {summary['std_specificity']:.4f}")

        return summary


def main():
    """Main entry point for training script."""
    logger.info("="*60)
    logger.info("Heart Disease Detection Model Training")
    logger.info("="*60)

    # Generate synthetic dataset for demonstration
    logger.info("Generating synthetic dataset...")
    np.random.seed(42)
    n_samples = 200
    ppg_signals = np.random.randn(n_samples, 1000)
    labels = np.random.randint(0, 2, n_samples)

    logger.info(f"Dataset: {n_samples} samples")
    logger.info(f"Label distribution: {np.bincount(labels)[0]} healthy, {np.bincount(labels)[1]} disease")

    # Configure model
    model_config = {
        "cnn_input_shape": (1000, 1),
        "hrv_input_shape": 8,
        "cnn_filters": [32, 64, 128],
        "cnn_kernel_sizes": [3, 3, 3],
        "dense_units": [128, 64],
        "l2_reg": 0.001,
        "dropout_rate": 0.3,
    }

    # Initialize training pipeline
    pipeline = TrainingPipeline(model_config=model_config, n_splits=3)

    # Train with cross-validation
    cv_summary = pipeline.train(
        ppg_signals=ppg_signals,
        labels=labels,
        epochs=50,
        validation_split=0.2,
    )

    logger.info("\nTraining completed successfully!")
    logger.info(f"Results saved to: {pipeline.log_dir}")


if __name__ == "__main__":
    main()
