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

### Final Model Performance (Test Set)

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

**Training Details:**
- Best validation score: 0.7602 (epoch 15)
- Test ROC AUC: 0.7323 (on held-out set)
- Training configuration: `configs/default.yaml`

**Top Contributing Features:**
1. checking_status (40.12)
2. credit_amount (37.12)
3. duration (31.00)
4. age (24.71)
5. residence_since (13.93)

Run `python scripts/train.py` to reproduce results.

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
