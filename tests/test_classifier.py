import numpy as np
import pandas as pd
import pytest
from sovereign_bond_classifier.classifier import SovereignKMeansClassifier


def sample_data():
    return pd.DataFrame({
        "a": [-4, -3.8, -4.2, 0, .2, -.2, 4, 4.2, 3.8],
        "b": [-3.9, -4.1, -3.8, .1, -.1, 0, 4.1, 3.9, 4.2],
        "risk": [.1, .12, .08, .5, .45, .55, .9, .95, .92],
        "default_event": [False] * 6 + [True] * 3,
    })


def test_fit_predict_and_mapping():
    data = sample_data()
    model = SovereignKMeansClassifier(["a", "b", "risk"], random_state=1).fit(data)
    raw = model.predict(data)
    assert set(raw.risk_label).issubset({"Cluster_0", "Cluster_1", "Cluster_2"})
    model.map_labels({0: "Safe", 1: "Risky", 2: "Default"})
    assert set(model.predict(data).risk_label).issubset({"Safe", "Risky", "Default"})


def test_default_mapping_requires_event_evidence():
    data = sample_data()
    data["default_event"] = False
    model = SovereignKMeansClassifier(["a", "b", "risk"], default_event_column="default_event", random_state=1)
    with pytest.raises(ValueError, match="no documented default event"):
        model.fit_and_map_from_risk_score(data, risk_score_column="risk")


def test_missing_feature_fails():
    with pytest.raises(ValueError, match="Missing feature columns"):
        SovereignKMeansClassifier(["missing"]).fit(sample_data())


def test_nonfinite_values_are_median_imputed():
    data = sample_data()
    data.loc[0, "a"] = np.inf
    model = SovereignKMeansClassifier(["a", "b", "risk"], random_state=1).fit(data)
    assert np.isfinite(model.centroids_scaled_).all()
