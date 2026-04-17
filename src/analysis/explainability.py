"""SHAP-based explainability for the hybrid heart disease model."""

from pathlib import Path
from typing import Optional, Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import shap
from tensorflow.keras import Model


def _ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_shap_arrays(shap_values):
    """Normalize SHAP output shape for multi-input models."""
    if isinstance(shap_values, list):
        # shap_values for binary output often returns [output0_values]
        shap_values = shap_values[0] if len(shap_values) == 1 else shap_values
    return shap_values


def explain_hybrid_model(
    model: Model,
    X_test: Tuple[np.ndarray, np.ndarray],
    background_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    hrv_feature_names: Optional[List[str]] = None,
    output_dir: str = "explainability",
    max_display: int = 20,
    nsamples: int = 100,
) -> Dict[str, str]:
    """
    Generate SHAP explainability plots for a hybrid model.

    Args:
        model: Trained Keras hybrid model with two inputs.
        X_test: Tuple of (ppg_signals, hrv_features) for the test set.
        background_data: Optional tuple of background examples for SHAP.
                         If None, the first 50 test samples are used.
        hrv_feature_names: Optional list of HRV feature names.
        output_dir: Directory to save summary plots.
        max_display: Maximum number of features to display in summary plots.
        nsamples: Number of SHAP samples to approximate values.

    Returns:
        Dictionary with file paths for saved explainability plots.
    """
    output_path = _ensure_output_dir(output_dir)

    ppg_test, hrv_test = X_test
    if background_data is None:
        background_data = (
            ppg_test[: min(50, len(ppg_test))],
            hrv_test[: min(50, len(hrv_test))],
        )

    explainer = shap.GradientExplainer(model, background_data)
    shap_values = explainer.shap_values([ppg_test, hrv_test], nsamples=nsamples)
    shap_values = _extract_shap_arrays(shap_values)

    if not isinstance(shap_values, list) or len(shap_values) != 2:
        raise ValueError(
            "SHAP values for hybrid model should return a list of two arrays "
            "(PPG branch and HRV branch)."
        )

    shap_ppg, shap_hrv = shap_values

    if hrv_feature_names is None:
        hrv_feature_names = [f"HRV_{i+1}" for i in range(hrv_test.shape[1])]

    plot_files: Dict[str, str] = {}

    # HRV feature importance summary plot
    hrvsum_path = output_path / "shap_summary_hrv.png"
    shap.summary_plot(
        shap_hrv,
        hrv_test,
        feature_names=hrv_feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(hrvsum_path, dpi=300)
    plt.close()
    plot_files["hrv_summary"] = str(hrvsum_path)

    # PPG signal importance summary plot
    shap_ppg_flat = shap_ppg.reshape(shap_ppg.shape[0], -1)
    ppg_feature_names = [f"PPG_{i}" for i in range(shap_ppg_flat.shape[1])]
    ppgsum_path = output_path / "shap_summary_ppg.png"
    shap.summary_plot(
        shap_ppg_flat,
        ppg_test.reshape(ppg_test.shape[0], -1),
        feature_names=ppg_feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(ppgsum_path, dpi=300)
    plt.close()
    plot_files["ppg_summary"] = str(ppgsum_path)

    return plot_files
