# SovereignBondClassifier

Python research prototype for structural identification and three-class K-Means clustering of sovereign-risk features.

## Status and interpretation

The implementation provides deterministic continuous-time bond valuation helpers, a common-component OLS model, and a three-cluster K-Means classifier. It does **not** establish that any sovereign is absolutely risk-free and does not estimate calibrated default probabilities. K-Means identifiers are arbitrary; `Safe`, `Risky`, and `Default` require an explicit, independently validated mapping.

## Install and test

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Quick start

```python
import pandas as pd
from sovereign_bond_classifier import SovereignKMeansClassifier

data = pd.read_csv("sovereign_features.csv")
features = ["pricing_deviation", "yield_volatility", "fiscal_stress", "liquidity_risk"]
model = SovereignKMeansClassifier(features, random_state=42, default_event_column="default_event")
model.fit(data)
print(model.cluster_summary())

# Only after independent validation, fill in the mapping supported by evidence.
# Cluster IDs vary across fitted models; this illustrative mapping is not universal.
model.map_labels({0: "Safe", 1: "Risky", 2: "Default"})
print(model.predict(data))
```

Alternatively, `fit_and_map_from_risk_score` orders clusters by a declared risk-score feature. When a default-event column is supplied, the top-risk cluster must contain a documented event before it can be labelled `Default`. This is only a guard, not complete validation.

## Continuous-time valuation

```python
from sovereign_bond_classifier import continuous_discount_factor, continuous_bond_price, continuous_duration
rate = lambda t: 0.04
coupon_flow = lambda t: 5.0
factor = continuous_discount_factor(0.0, 5.0, rate)
price = continuous_bond_price(0.0, 5.0, coupon_flow, 100.0, rate)
duration = continuous_duration(0.0, 5.0, coupon_flow, 100.0, rate)
```

Rates are continuously compounded annual rates and times are in years. These helper functions are deterministic discounted-cash-flow calculations, not a complete stochastic defaultable-bond model.

## Structural model

```python
from sovereign_bond_classifier import CommonComponentModel
model = CommonComponentModel().fit(panel, target="rho", features=["z1", "z2"])
print(model.common_component_)
print(model.results_)
```

The fitted common component is statistical, not automatically a risk-free rate. The model checks whether the intercept can be separately identified from the chosen regressors.

## Data and validation

A serious empirical dataset should contain sovereign and security identifiers, observation date, currency, maturity, prices and cash-flow terms, fiscal/debt-service indicators, liquidity and volatility measures, structural-model features, and a documented default-event source and definition. Do not compare unmatched currencies or maturities without explicit controls. Fit preprocessing on training data only and use time-aware validation to avoid look-ahead leakage. A high yield alone does not establish default.

The example in `examples/demo_synthetic.py` uses invented data only and is not evidence about real countries. See `LICENSE` for licensing.
