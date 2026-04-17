"""Hybrid TensorFlow model for heart disease classification.

This module implements a hybrid deep learning architecture combining 1D-CNN
for PPG signal processing and MLP for statistical HRV features.
"""

from typing import Tuple, Dict, Optional, List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers


class HybridHeartDiseaseClassifier:
    """
    Hybrid neural network for heart disease classification.

    Combines two parallel branches:
    - 1D-CNN branch: Processes raw PPG signals (time-series data)
    - MLP branch: Processes statistical HRV features

    Both branches are concatenated and fed through dense layers
    for final binary classification (healthy vs. disease).

    Attributes:
        cnn_input_shape (Tuple[int, int]): Shape of PPG signal input (samples, channels).
        hrv_input_shape (Tuple[int]): Shape of HRV features input (num_features,).
        cnn_filters (List[int]): Number of filters for each Conv1D layer.
        cnn_kernel_sizes (List[int]): Kernel sizes for each Conv1D layer.
        dense_units (List[int]): Units for dense layers in both branches.
        l2_reg (float): L2 regularization strength.
        dropout_rate (float): Dropout rate for regularization.
    """

    def __init__(
        self,
        cnn_input_shape: Tuple[int, int] = (1000, 1),
        hrv_input_shape: int = 8,
        cnn_filters: Optional[List[int]] = None,
        cnn_kernel_sizes: Optional[List[int]] = None,
        dense_units: Optional[List[int]] = None,
        l2_reg: float = 0.001,
        dropout_rate: float = 0.3,
    ) -> None:
        """
        Initialize the Hybrid Classifier.

        Args:
            cnn_input_shape: Shape of PPG signal input (samples, channels).
                           Default is (1000, 1) for 10s at 100Hz.
            hrv_input_shape: Number of HRV features. Default is 8.
            cnn_filters: List of filter counts for Conv1D layers.
                       Default is [32, 64, 128].
            cnn_kernel_sizes: List of kernel sizes for Conv1D layers.
                            Default is [3, 3, 3].
            dense_units: List of units for dense layers in both branches.
                       Default is [128, 64].
            l2_reg: L2 regularization strength. Default is 0.001.
            dropout_rate: Dropout rate. Default is 0.3.

        Raises:
            ValueError: If input shapes are invalid or filter/kernel lists don't match.
        """
        if cnn_input_shape[0] <= 0 or cnn_input_shape[1] <= 0:
            raise ValueError("cnn_input_shape dimensions must be positive")
        if hrv_input_shape <= 0:
            raise ValueError("hrv_input_shape must be positive")
        if not (0 <= dropout_rate < 1):
            raise ValueError("dropout_rate must be between 0 and 1")
        if l2_reg < 0:
            raise ValueError("l2_reg must be non-negative")

        self.cnn_input_shape = cnn_input_shape
        self.hrv_input_shape = hrv_input_shape
        self.cnn_filters = cnn_filters or [32, 64, 128]
        self.cnn_kernel_sizes = cnn_kernel_sizes or [3, 3, 3]
        self.dense_units = dense_units or [128, 64]
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate

        if len(self.cnn_filters) != len(self.cnn_kernel_sizes):
            raise ValueError(
                "cnn_filters and cnn_kernel_sizes must have the same length"
            )

        self.model: Optional[Model] = None

    def build_cnn_branch(self, cnn_input: layers.Input) -> layers.Layer:
        """
        Build the 1D-CNN branch for PPG signal processing.

        Architecture:
        - Multiple Conv1D layers with ReLU activation
        - MaxPooling1D after each Conv1D
        - Batch normalization for training stability
        - Dropout for regularization

        Args:
            cnn_input: Input layer for PPG signal.

        Returns:
            Output tensor from CNN branch.
        """
        x = cnn_input

        # Stack Conv1D layers with pooling
        for filters, kernel_size in zip(self.cnn_filters, self.cnn_kernel_sizes):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(self.l2_reg),
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Global average pooling to reduce dimensionality
        x = layers.GlobalAveragePooling1D()(x)

        return x

    def build_mlp_branch(self, hrv_input: layers.Input) -> layers.Layer:
        """
        Build the MLP branch for HRV features processing.

        Architecture:
        - Multiple dense layers with ReLU activation
        - Batch normalization for training stability
        - Dropout for regularization
        - L2 regularization on weights

        Args:
            hrv_input: Input layer for HRV features.

        Returns:
            Output tensor from MLP branch.
        """
        x = hrv_input

        # Stack dense layers
        for units in self.dense_units:
            x = layers.Dense(
                units=units,
                activation="relu",
                kernel_regularizer=regularizers.l2(self.l2_reg),
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        return x

    def build_model(self) -> Model:
        """
        Build the complete hybrid model.

        Architecture:
        1. Create two input layers (PPG signal and HRV features)
        2. Build CNN branch for signal processing
        3. Build MLP branch for feature processing
        4. Concatenate both branches
        5. Add final dense layers with dropout
        6. Output layer with Sigmoid activation for binary classification

        Returns:
            Compiled Keras Model.
        """
        # Input layers
        cnn_input = layers.Input(shape=self.cnn_input_shape, name="ppg_signal")
        hrv_input = layers.Input(shape=(self.hrv_input_shape,), name="hrv_features")

        # Build branches
        cnn_branch = self.build_cnn_branch(cnn_input)
        mlp_branch = self.build_mlp_branch(hrv_input)

        # Concatenate branches
        merged = layers.Concatenate()([cnn_branch, mlp_branch])

        # Final classification layers
        x = layers.Dense(
            units=64,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.l2_reg),
        )(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Dense(
            units=32,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.l2_reg),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Output layer with Sigmoid for binary classification
        output = layers.Dense(
            units=1,
            activation="sigmoid",
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="disease_probability",
        )(x)

        # Create model
        self.model = Model(inputs=[cnn_input, hrv_input], outputs=output)

        return self.model

    def compile_model(
        self,
        optimizer: Optional[str] = "adam",
        loss: Optional[str] = "binary_crossentropy",
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Compile the model with optimizer, loss, and metrics.

        Args:
            optimizer: Optimizer to use. Default is "adam".
            loss: Loss function. Default is "binary_crossentropy".
            metrics: List of metrics to track. Default is ["accuracy", "auc"].

        Raises:
            RuntimeError: If model has not been built yet.
        """
        if self.model is None:
            raise RuntimeError("Model must be built before compilation. Call build_model() first.")

        if metrics is None:
            metrics = ["accuracy", "auc"]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_model(self) -> Model:
        """
        Get the compiled model.

        Returns:
            Keras Model object.

        Raises:
            RuntimeError: If model has not been built yet.
        """
        if self.model is None:
            raise RuntimeError("Model not built yet. Call build_model() first.")
        return self.model

    def summary(self) -> None:
        """Print model architecture summary."""
        if self.model is None:
            raise RuntimeError("Model not built yet. Call build_model() first.")
        self.model.summary()

    def get_config(self) -> Dict:
        """
        Get configuration dictionary of the model.

        Returns:
            Dictionary with model configuration.
        """
        return {
            "cnn_input_shape": self.cnn_input_shape,
            "hrv_input_shape": self.hrv_input_shape,
            "cnn_filters": self.cnn_filters,
            "cnn_kernel_sizes": self.cnn_kernel_sizes,
            "dense_units": self.dense_units,
            "l2_reg": self.l2_reg,
            "dropout_rate": self.dropout_rate,
        }


def create_hybrid_classifier(
    cnn_input_shape: Tuple[int, int] = (1000, 1),
    hrv_input_shape: int = 8,
    cnn_filters: Optional[List[int]] = None,
    cnn_kernel_sizes: Optional[List[int]] = None,
    dense_units: Optional[List[int]] = None,
    l2_reg: float = 0.001,
    dropout_rate: float = 0.3,
    compile: bool = True,
) -> Model:
    """
    Factory function to create and optionally compile a hybrid classifier model.

    This is a convenience function for quick model creation.

    Args:
        cnn_input_shape: Shape of PPG signal input. Default is (1000, 1).
        hrv_input_shape: Number of HRV features. Default is 8.
        cnn_filters: List of CNN filter counts. Default is [32, 64, 128].
        cnn_kernel_sizes: List of CNN kernel sizes. Default is [3, 3, 3].
        dense_units: List of dense layer units. Default is [128, 64].
        l2_reg: L2 regularization strength. Default is 0.001.
        dropout_rate: Dropout rate. Default is 0.3.
        compile: Whether to compile the model. Default is True.

    Returns:
        Compiled (or uncompiled) Keras Model.

    Example:
        >>> model = create_hybrid_classifier()
        >>> model.summary()
    """
    classifier = HybridHeartDiseaseClassifier(
        cnn_input_shape=cnn_input_shape,
        hrv_input_shape=hrv_input_shape,
        cnn_filters=cnn_filters,
        cnn_kernel_sizes=cnn_kernel_sizes,
        dense_units=dense_units,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
    )

    model = classifier.build_model()

    if compile:
        classifier.compile_model()

    return model


class ModelTrainer:
    """Utility class for training the hybrid classifier with early stopping."""

    def __init__(self, model: Model, patience: int = 10) -> None:
        """
        Initialize the trainer.

        Args:
            model: Compiled Keras model.
            patience: Number of epochs with no improvement before stopping.
        """
        self.model = model
        self.patience = patience
        self.history = None

    def train(
        self,
        X_train: Tuple[np.ndarray, np.ndarray],
        y_train: np.ndarray,
        X_val: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict:
        """
        Train the model with optional early stopping.

        Args:
            X_train: Tuple of (PPG_signals, HRV_features) for training.
            y_train: Training labels (0 or 1).
            X_val: Tuple of (PPG_signals, HRV_features) for validation.
            y_val: Validation labels.
            epochs: Maximum number of epochs. Default is 100.
            batch_size: Batch size. Default is 32.
            verbose: Verbosity level. Default is 1.

        Returns:
            Training history dictionary.
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        return self.history.history

    def predict(
        self, X: Tuple[np.ndarray, np.ndarray], threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.

        Args:
            X: Tuple of (PPG_signals, HRV_features).
            threshold: Classification threshold. Default is 0.5.

        Returns:
            Tuple of (probabilities, binary_predictions).
        """
        probabilities = self.model.predict(X, verbose=0)
        predictions = (probabilities > threshold).astype(int).flatten()
        return probabilities.flatten(), predictions

    def evaluate(
        self, X_test: Tuple[np.ndarray, np.ndarray], y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            X_test: Tuple of (PPG_signals, HRV_features) for testing.
            y_test: Test labels.

        Returns:
            Dictionary with loss and metrics.
        """
        loss, *metrics = self.model.evaluate(X_test, y_test, verbose=0)
        metric_names = self.model.metrics_names[1:]
        return {"loss": loss, **dict(zip(metric_names, metrics))}
