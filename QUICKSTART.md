# Quick Start Guide

This guide will get you up and running with the Adaptive Feature Importance Reweighting project in under 5 minutes.

## Installation

```bash
# Clone or navigate to the project directory
cd adaptive-feature-importance-reweighting-credit-risk

# Install dependencies
pip install -r requirements.txt
```

## Run Your First Training

```bash
# Train the model with default configuration (full adaptive reweighting)
python scripts/train.py --config configs/default.yaml

# This will:
# - Generate synthetic Home Credit-like data (if real data not available)
# - Train an ensemble of LightGBM, XGBoost, and CatBoost
# - Apply adaptive feature importance reweighting
# - Save the best model to models/best_model.pkl
# - Output results to results/training_results.json
```

## Evaluate the Trained Model

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint models/best_model.pkl

# This will:
# - Load the trained model
# - Generate predictions on test data
# - Compute comprehensive metrics (ROC-AUC, PR-AUC, KS statistic, etc.)
# - Create evaluation plots in results/
# - Save detailed results to results/evaluation.json
```

## Make Predictions

```bash
# Create sample input data (for demonstration)
# In production, you would use your own data file

# Make predictions
python scripts/predict.py \
    --checkpoint models/best_model.pkl \
    --input data/sample.csv \
    --output predictions.csv

# This will output predictions with confidence scores and risk levels
```

## Run Ablation Study

Compare baseline (no adaptive reweighting) vs. full model:

```bash
# Train baseline model (no adaptive features)
python scripts/train.py --config configs/ablation.yaml

# Train full model (with adaptive reweighting)
python scripts/train.py --config configs/default.yaml

# Compare results in results/ directory
```

## Run Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS
# or
xdg-open htmlcov/index.html  # On Linux
```

## Quick Training with Small Data

For quick testing with reduced data and epochs:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --epochs 10 \
    --seed 42
```

## Understanding the Configuration

The configuration files control all aspects of training:

- `configs/default.yaml`: Full model with adaptive reweighting enabled
- `configs/ablation.yaml`: Baseline model without adaptive features

Key configuration sections:
- `model`: Ensemble weights and hyperparameters
- `reweighting`: Adaptive feature importance settings
- `curriculum`: Curriculum learning schedule
- `training`: Training loop parameters
- `metrics`: Evaluation metrics to track

## Common Issues

### Missing Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Import Errors

Make sure you're running scripts from the project root:

```bash
cd /path/to/adaptive-feature-importance-reweighting-credit-risk
python scripts/train.py
```

### GPU/CPU Selection

The models automatically detect and use available hardware. For tree-based models (LightGBM, XGBoost, CatBoost), CPU is often sufficient and fast.

## Next Steps

1. **Customize hyperparameters**: Edit `configs/default.yaml`
2. **Add your own data**: Place CSV in `data/` and modify `data.dataset_name` in config
3. **Extend the model**: Add new components in `src/models/components.py`
4. **Track experiments**: MLflow UI will be available at `mlruns/`

## Key Files to Explore

- `src/models/components.py`: Custom adaptive reweighting and curriculum learning
- `src/training/trainer.py`: Main training loop with progressive reweighting
- `src/evaluation/metrics.py`: Comprehensive credit risk metrics
- `configs/default.yaml`: Full configuration reference

## Getting Help

For detailed documentation, see:
- `README.md`: Project overview and usage
- `PROJECT_SUMMARY.txt`: Detailed project structure and features
- `REQUIREMENTS_CHECKLIST.md`: Complete requirements verification

## License

MIT License - Copyright (c) 2026 Alireza Shojaei
