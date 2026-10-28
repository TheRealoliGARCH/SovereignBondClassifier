"""Synthetic software demonstration; these are not real sovereign observations."""
import pandas as pd
from sovereign_bond_classifier import SovereignKMeansClassifier


def main():
    data = pd.DataFrame({
        "entity": [f"Entity_{i:02d}" for i in range(18)],
        "pricing_deviation": [-.8,-.7,-.6,-.9,-.5,-.4, 0,.1,.2,0,.3,.4, .9,1,1.1,1.2,1,1.3],
        "yield_volatility": [.1,.12,.11,.09,.15,.14, .3,.32,.29,.35,.31,.34, .8,.9,.85,1,.95,1.1],
        "fiscal_stress": [.1,.15,.12,.1,.18,.2, .35,.4,.3,.45,.38,.42, .7,.8,.75,.85,.82,.9],
        "liquidity_risk": [.1,.12,.09,.14,.11,.15, .3,.35,.32,.4,.38,.33, .7,.8,.75,.85,.9,.82],
        "default_event": [False]*12 + [True]*6,
    })
    features = ["pricing_deviation", "yield_volatility", "fiscal_stress", "liquidity_risk"]
    model = SovereignKMeansClassifier(features, default_event_column="default_event", random_state=7)
    model.fit_and_map_from_risk_score(data, risk_score_column="fiscal_stress")
    print(model.cluster_summary())
    print(model.predict(data[["entity", *features]]).to_string(index=False))


if __name__ == "__main__":
    main()
