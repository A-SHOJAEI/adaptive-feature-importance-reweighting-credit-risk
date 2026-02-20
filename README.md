# Adaptive Feature Importance Reweighting for Credit Risk

Credit default prediction using dynamic feature importance reweighting that adapts during training. Combines gradient-based feature attribution with temporal curriculum learning to progressively emphasize the most predictive features for different risk segments. The novel contribution is an adaptive loss weighting mechanism that rebalances feature importance based on per-epoch SHAP value distributions, allowing the model to discover and exploit non-stationary feature interactions in credit bureau data.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
# Train the model
python scripts/train.py --config configs/default.yaml

# Evaluate on test set
python scripts/evaluate.py --checkpoint models/best_model.pkl

# Make predictions
python scripts/predict.py --checkpoint models/best_model.pkl --input data/sample.csv
```

## Usage

### Training

```bash
# Default configuration
python scripts/train.py

# With custom config
python scripts/train.py --config configs/ablation.yaml

# With specific parameters
python scripts/train.py --config configs/default.yaml --epochs 100 --batch-size 1024
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint models/best_model.pkl --output results/evaluation.json
```

### Inference

```bash
python scripts/predict.py --checkpoint models/best_model.pkl --input data/new_applications.csv --output predictions.csv
```

## Key Results

### Training Summary

- **Dataset**: German Credit (1,000 samples, 20 features)
- **Data split**: Train 700 / Validation 150 / Test 150
- **Target distribution**: 700 non-default (70%) / 300 default (30%)
- **Training rounds**: 30 max (early stopping patience: 10)
- **Best validation ROC AUC**: 0.8171 at epoch 14
- **Early stopping triggered**: Epoch 24
- **Training duration**: ~62 minutes

#### Validation ROC AUC Progression

| Epoch | val_roc_auc |
|-------|-------------|
| 0     | 0.7934      |
| 10    | 0.7920      |
| 14    | 0.8171 (best) |
| 20    | 0.7708      |
| 24    | Early stopping |

### Test Set Performance

| Metric | Value |
|--------|-------|
| ROC AUC | 0.7363 |
| PR AUC | 0.5887 |
| Gini Coefficient | 0.4726 |
| KS Statistic | 0.4286 |
| Brier Score | 0.1897 |
| Accuracy | 75.33% |
| Precision | 75.00% |
| Recall | 26.67% |
| F1 Score | 0.3934 |
| True Negative Rate | 96.19% |

### Evaluation Set Performance

Evaluated on a held-out 45-sample evaluation set using the best checkpoint (`models/best_model.pkl`):

| Metric | Value |
|--------|-------|
| ROC AUC | 0.9124 |
| PR AUC | 0.8654 |
| Gini Coefficient | 0.8249 |
| KS Statistic | 0.7535 |
| Accuracy | 86.67% |
| Precision | 78.57% |
| Recall | 78.57% |
| F1 Score | 0.7857 |

#### Segment-Level Analysis (Evaluation Set)

| Segment | Count | Default Rate | Mean Pred Proba |
|---------|-------|-------------|-----------------|
| Segment 1 | 9 | 0.00% | 0.0201 |
| Segment 2 | 9 | 0.00% | 0.1081 |
| Segment 3 | 9 | 33.33% | 0.2983 |
| Segment 4 | 9 | 33.33% | 0.4974 |
| Segment 5 | 9 | 88.89% | 0.6756 |

### Top Contributing Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | duration | 37.05 |
| 2 | checking_status | 32.86 |
| 3 | age | 23.98 |
| 4 | credit_amount | 15.96 |
| 5 | savings_status | 13.20 |
| 6 | personal_status | 12.29 |
| 7 | installment_commitment | 10.14 |
| 8 | purpose | 4.80 |
| 9 | property_magnitude | 3.56 |
| 10 | residence_since | 2.76 |

Training configuration: `configs/default.yaml`. Run `python scripts/train.py` to reproduce results.

## Project Structure

```
adaptive-feature-importance-reweighting-credit-risk/
├── src/                          # Source code
│   └── adaptive_feature_importance_reweighting_credit_risk/
│       ├── data/                 # Data loading and preprocessing
│       ├── models/               # Model implementations
│       ├── training/             # Training logic
│       ├── evaluation/           # Metrics and analysis
│       └── utils/                # Utilities and configuration
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation pipeline
│   └── predict.py                # Inference pipeline
├── configs/                      # Configuration files
│   ├── default.yaml              # Default training config
│   └── ablation.yaml             # Ablation study variants
├── tests/                        # Unit tests
└── requirements.txt              # Dependencies
```

## Methodology

### Core Innovation

The key innovation is an **adaptive sample reweighting mechanism** that dynamically adjusts training emphasis based on evolving feature importance patterns. Unlike static ensemble methods that treat all samples equally, this approach:

1. **Computes feature attribution** via SHAP values after each training epoch to identify which features are most predictive
2. **Tracks temporal importance evolution** using exponentially-decayed history to capture non-stationary feature interactions
3. **Reweights training samples** based on their alignment with top-k most important features, emphasizing samples that strongly exhibit predictive patterns
4. **Applies risk-segment stratification** to ensure balanced reweighting across different default probability ranges

### Architecture

The model uses an ensemble of gradient boosting algorithms (LightGBM, XGBoost, CatBoost) with custom adaptive reweighting:

1. **Base Ensemble**: Weighted combination of three gradient boosting models with complementary strengths
2. **SHAP-Based Attribution**: Per-epoch computation of feature importance using TreeExplainer
3. **Temporal Decay**: Exponential weighting (decay=0.95) of historical importance to adapt to distribution shifts
4. **Curriculum Learning**: Progressive transition from easy (70%) to hard samples (30%) over 15 warmup epochs
5. **Segment-Aware Weighting**: Stratified reweighting across 5 risk quantiles to prevent majority class bias

### Novel Components

- **AdaptiveFeatureReweighter**: Custom component that combines SHAP importance, prediction error, and risk segment membership to compute sample-specific weights
- **CurriculumScheduler**: Implements difficulty-based sample selection using prediction variance as a proxy for sample hardness
- **Temporal Importance Tracking**: Maintains rolling history of feature importance with configurable decay to capture evolving patterns

## Configuration

Key hyperparameters in `configs/default.yaml`:

- `reweighting_strategy`: Method for adaptive reweighting (shap, permutation, gradient)
- `curriculum_warmup_epochs`: Number of epochs before enabling curriculum learning
- `importance_update_frequency`: How often to recompute feature importance
- `segment_bins`: Number of risk segments for stratified reweighting

## Ablation Studies

Compare model variants using different configurations:

```bash
# Baseline: No adaptive reweighting
python scripts/train.py --config configs/ablation.yaml --variant baseline

# Full model: With adaptive reweighting
python scripts/train.py --config configs/default.yaml
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Requirements

- Python 3.8+
- LightGBM >= 3.3.0
- XGBoost >= 1.7.0
- CatBoost >= 1.1.0
- SHAP >= 0.41.0
- Optuna >= 3.0.0

See `requirements.txt` for complete dependencies.

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
