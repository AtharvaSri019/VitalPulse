# Heart Disease Detection ML Pipeline

A professional machine learning pipeline for detecting heart disease using clinical and physiological data.

## Project Structure

```
heart_disease_detection/
├── data/
│   └── raw/                 # Raw input data
├── src/
│   ├── preprocessing/       # Data cleaning and preprocessing
│   ├── features/            # Feature engineering and extraction
│   ├── models/              # Model training and evaluation
│   └── utils/               # Utility functions and helpers
├── tests/                   # Unit and integration tests
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd heart_disease_detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- **TensorFlow**: Deep learning framework for model development
- **SciPy**: Scientific computing tools
- **NeuroKit2**: Cardiac signal processing and heart rate analysis
- **scikit-learn**: Machine learning utilities and preprocessing
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization
- **numpy**: Numerical computing
- **pytest**: Testing framework

## Usage

```python
from src.preprocessing import load_data, clean_data
from src.features import extract_features, scale_features
from src.models import train_model, evaluate_model

# Load and preprocess data
X, y = load_data("data/raw/dataset.csv")
X_clean = clean_data(X)

# Feature engineering
X_features = extract_features(X_clean)
X_scaled = scale_features(X_features)

# Train and evaluate model
model = train_model(X_scaled, y)
metrics = evaluate_model(model, X_scaled, y)
```

## Model Development

This pipeline follows best practices for medical ML:
- Modular architecture for easy maintenance
- Proper separation of concerns
- Comprehensive logging and error handling
- Unit tests for each component
- Configuration management

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## License

[Add your license information]

## Contact

[Add contact information]
