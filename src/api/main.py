"""FastAPI application for heart disease risk prediction."""

from pathlib import Path
from typing import Annotated, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing.signal_cleaner import PPGProcessor
from src.features.hrv_metrics import HRVMetrics
from src.models.classifier import create_hybrid_classifier


class PPGRequest(BaseModel):
    ppg_signal: Annotated[List[float], Field(min_length=10)]
    sample_rate: Optional[float] = 100.0


class PredictionResponse(BaseModel):
    probability: float
    label: int
    hrv_features: Dict[str, float]


app = FastAPI(
    title="Heart Disease Detection API",
    version="0.1.0",
    description="Predict heart disease risk from PPG signal segments.",
)


MODEL_PATH = Path("checkpoints/latest_model.h5")
SAMPLE_RATE = 100.0
WINDOW_LENGTH = 1000


def load_model() -> any:
    model = create_hybrid_classifier(compile=False)
    if MODEL_PATH.exists():
        model.load_weights(str(MODEL_PATH))
    return model


@app.on_event("startup")
def startup_event() -> None:
    """Initialize processing pipeline and model when the app starts."""
    app.state.ppg_processor = PPGProcessor(sample_rate=SAMPLE_RATE)
    app.state.hrv_extractor = HRVMetrics(sample_rate=SAMPLE_RATE)
    app.state.model = load_model()


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PPGRequest) -> PredictionResponse:
    """Predict the risk probability for incoming PPG signal data."""
    ppg_signal = np.array(payload.ppg_signal, dtype=np.float32)

    if ppg_signal.ndim != 1:
        raise HTTPException(status_code=400, detail="ppg_signal must be a 1D array")

    sample_rate = payload.sample_rate
    if sample_rate <= 0:
        raise HTTPException(status_code=400, detail="sample_rate must be positive")

    if ppg_signal.size < 10:
        raise HTTPException(status_code=400, detail="ppg_signal must contain at least 10 samples")

    if ppg_signal.size != WINDOW_LENGTH:
        if ppg_signal.size < WINDOW_LENGTH:
            padded = np.zeros(WINDOW_LENGTH, dtype=np.float32)
            padded[: ppg_signal.size] = ppg_signal
            ppg_signal = padded
        else:
            ppg_signal = ppg_signal[:WINDOW_LENGTH]

    processor: PPGProcessor = app.state.ppg_processor
    if processor.sample_rate != sample_rate:
        processor = PPGProcessor(sample_rate=sample_rate)
        app.state.ppg_processor = processor

    try:
        cleaned_signal, _ = processor.process_signal(ppg_signal, artifact_method="moving_average")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    hrv_extractor: HRVMetrics = app.state.hrv_extractor
    if hrv_extractor.sample_rate != sample_rate:
        hrv_extractor = HRVMetrics(sample_rate=sample_rate)
        app.state.hrv_extractor = hrv_extractor

    try:
        hrv_metrics = hrv_extractor.extract_all_hrv_features(cleaned_signal)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    hrv_vector = np.array(
        [
            hrv_metrics["SDNN_ms"],
            hrv_metrics["RMSSD_ms"],
            hrv_metrics["mean_HR_bpm"],
            hrv_metrics["LF"],
            hrv_metrics["HF"],
            hrv_metrics["LF_HF_ratio"],
            hrv_metrics["LF_norm"],
            hrv_metrics["HF_norm"],
        ],
        dtype=np.float32,
    )

    ppg_input = cleaned_signal.reshape(1, -1, 1)
    hrv_input = hrv_vector.reshape(1, -1)

    try:
        probability = float(app.state.model.predict([ppg_input, hrv_input], verbose=0)[0, 0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictionResponse(
        probability=probability,
        label=int(probability >= 0.5),
        hrv_features=hrv_metrics,
    )
