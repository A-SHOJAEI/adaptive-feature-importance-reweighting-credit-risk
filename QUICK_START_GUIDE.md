# Quick Start Guide

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Train the Model

```bash
# Train with default configuration (includes adaptive reweighting)
python scripts/train.py --config configs/default.yaml

# Train baseline (no adaptive mechanisms) for comparison
python scripts/train.py --config configs/ablation.yaml

# Train with custom parameters
python scripts/train.py --config configs/default.yaml --epochs 50 --seed 123
```

**Output**:
- Trained model: `models/best_model.pkl`
- Training logs: `logs/train.log`
- Results: `results/training_results.json`
- Feature importance: `results/feature_importance.csv`
- Plots: `results/*.png`

### 2. Evaluate the Model

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint models/best_model.pkl

# With custom output location
python scripts/evaluate.py \
  --checkpoint models/best_model.pkl \
  --output results/my_evaluation.json \
  --output-dir results/
```

**Output**:
- Evaluation metrics: `results/evaluation.json`
- Predictions: `results/predictions.csv`
- Analysis plots: `results/*.png`

**Metrics Computed**:
- ROC AUC, PR AUC, KS Statistic, Gini, Brier Score
- Accuracy, Precision, Recall, F1 Score
- TPR, TNR, FPR, FNR
- Segment-wise performance (5 risk quantiles)

### 3. Make Predictions

```bash
# Predict on new data
python scripts/predict.py \
  --checkpoint models/best_model.pkl \
  --input data/new_applications.csv \
  --output predictions.csv

# With custom threshold
python scripts/predict.py \
  --checkpoint models/best_model.pkl \
  --input data/new_applications.csv \
  --output predictions.csv \
  --threshold 0.3
```

**Input Format**: CSV/Parquet/JSON with feature columns

**Output**:
```csv
predicted_label,predicted_probability,default_risk
0,0.1234,Low
1,0.8765,Very High
0,0.3456,Medium
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Key parameters
reweighting:
  enabled: true              # Enable adaptive reweighting
  strategy: "shap"          # shap, permutation, or gradient
  top_k_features: 20        # Focus on top K features
  reweight_alpha: 0.3       # Reweighting strength (0-1)
  temporal_decay: 0.95      # Historical importance decay

curriculum:
  enabled: true             # Enable curriculum learning
  warmup_epochs: 15         # Warmup before curriculum
  initial_easy_ratio: 0.7   # Start with 70% easy samples
  final_easy_ratio: 0.3     # End with 30% easy samples

model:
  type: "ensemble"          # ensemble, lightgbm, xgboost, catboost
  ensemble_weights:
    lightgbm: 0.4
    xgboost: 0.35
    catboost: 0.25
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

## Ablation Study

Compare baseline vs. adaptive reweighting:

```bash
# 1. Train baseline (no adaptive mechanisms)
python scripts/train.py --config configs/ablation.yaml

# 2. Train full model (with adaptive mechanisms)
python scripts/train.py --config configs/default.yaml

# 3. Compare results
cat results/training_results.json
```

## MLflow Tracking

View experiment tracking:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Open browser to http://localhost:5000
```

## Project Structure

```
adaptive-feature-importance-reweighting-credit-risk/
├── configs/
│   ├── default.yaml        # Full model configuration
│   └── ablation.yaml       # Baseline configuration
├── scripts/
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Evaluation pipeline
│   └── predict.py          # Inference pipeline
├── src/adaptive_feature_importance_reweighting_credit_risk/
│   ├── data/               # Data loading & preprocessing
│   ├── models/             # Model & custom components
│   ├── training/           # Adaptive trainer
│   ├── evaluation/         # Metrics & analysis
│   └── utils/              # Utilities
├── tests/                  # Test suite
├── models/                 # Saved models
├── results/                # Outputs
└── logs/                   # Training logs
```

## Troubleshooting

### Import Errors
```bash
# Ensure src/ is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Memory Issues
```yaml
# In config file, reduce sample size
data:
  sample_frac: 0.3  # Use 30% of data
```

### Slow SHAP Computation
```yaml
# Reduce SHAP sample size or use gradient importance
reweighting:
  strategy: "gradient"  # Faster than SHAP
```

## Key Features

✅ **Adaptive Reweighting**: Dynamic sample weighting based on SHAP importance
✅ **Curriculum Learning**: Progressive easy-to-hard sample selection
✅ **Ensemble Models**: LightGBM + XGBoost + CatBoost
✅ **Comprehensive Metrics**: 13+ evaluation metrics
✅ **Risk Segmentation**: Performance analysis across risk quantiles
✅ **MLflow Integration**: Experiment tracking and visualization
✅ **Flexible Configuration**: YAML-based parameter management

## Next Steps

1. **Train the model**: `python scripts/train.py`
2. **Review results**: Check `results/training_results.json`
3. **Evaluate performance**: `python scripts/evaluate.py --checkpoint models/best_model.pkl`
4. **Make predictions**: `python scripts/predict.py --checkpoint models/best_model.pkl --input your_data.csv`
5. **Run ablation study**: Compare `configs/default.yaml` vs `configs/ablation.yaml`
