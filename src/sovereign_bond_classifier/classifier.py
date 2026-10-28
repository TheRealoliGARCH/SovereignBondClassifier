"""Three-cluster K-Means with explicit semantic label mapping."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

LABELS = ("Safe", "Risky", "Default")


class SovereignKMeansClassifier:
    """K-Means risk clustering; cluster IDs are not intrinsically Safe/Risky/Default."""
    def __init__(self, features: Sequence[str], *, n_clusters: int = 3,
                 random_state: int = 42, n_init: int = 20,
                 default_event_column: str | None = None) -> None:
        if n_clusters != 3:
            raise ValueError("This classifier requires exactly three clusters.")
        if not features or len(set(features)) != len(features):
            raise ValueError("Provide a non-empty list of unique features.")
        self.features = list(features)
        self.n_clusters, self.random_state, self.n_init = n_clusters, random_state, n_init
        self.default_event_column = default_event_column
        self.cluster_to_label_: dict[int, str] = {}

    def _prepare(self, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        x = frame[self.features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        if fit:
            self.imputer_ = SimpleImputer(strategy="median")
            self.scaler_ = StandardScaler()
            x = self.imputer_.fit_transform(x)
            if x.shape[0] < 3:
                raise ValueError("At least three observations are needed.")
            x = self.scaler_.fit_transform(x)
        else:
            if not hasattr(self, "scaler_"):
                raise RuntimeError("Call fit before predict.")
            x = self.scaler_.transform(self.imputer_.transform(x))
        if not np.all(np.isfinite(x)):
            raise ValueError("Preprocessing produced non-finite values; inspect features.")
        return x

    def fit(self, frame: pd.DataFrame) -> "SovereignKMeansClassifier":
        if self.default_event_column and self.default_event_column not in frame:
            raise ValueError(f"Missing default event column: {self.default_event_column}")
        x = self._prepare(frame, fit=True)
        self.kmeans_ = KMeans(n_clusters=3, random_state=self.random_state, n_init=self.n_init)
        self.cluster_ids_ = self.kmeans_.fit_predict(x)
        if len(np.unique(self.cluster_ids_)) != 3:
            raise ValueError("K-Means produced fewer than three occupied clusters.")
        self.centroids_scaled_ = self.kmeans_.cluster_centers_
        self.inertia_ = float(self.kmeans_.inertia_)
        self.silhouette_ = float(silhouette_score(x, self.cluster_ids_)) if len(frame) > 3 else float("nan")
        self.training_ = frame.copy()
        self.training_["cluster_id"] = self.cluster_ids_
        self.cluster_to_label_ = {}
        return self

    def map_labels(self, mapping: Mapping[int, str]) -> "SovereignKMeansClassifier":
        mapping = {int(k): str(v) for k, v in mapping.items()}
        if set(mapping) != {0, 1, 2} or set(mapping.values()) != set(LABELS):
            raise ValueError("Mapping must map cluster IDs 0, 1, 2 one-to-one to Safe, Risky, Default.")
        self.cluster_to_label_ = mapping
        return self

    def fit_and_map_from_risk_score(self, frame: pd.DataFrame, *, risk_score_column: str,
                                    default_event_column: str | None = None) -> "SovereignKMeansClassifier":
        """Order clusters by a risk score; Default requires a recorded event in the top cluster."""
        if risk_score_column not in self.features:
            raise ValueError("risk_score_column must be one of the clustering features.")
        event_col = default_event_column or self.default_event_column
        if event_col and event_col not in frame:
            raise ValueError(f"Missing default event column: {event_col}")
        self.fit(frame)
        scores = frame[risk_score_column].to_numpy(dtype=float)
        means = {k: float(np.nanmean(scores[self.cluster_ids_ == k])) for k in range(3)}
        ordered = sorted(means, key=means.get)
        mapping = {ordered[0]: "Safe", ordered[1]: "Risky", ordered[2]: "Default"}
        if event_col:
            events = frame[event_col].fillna(False).astype(bool).to_numpy()
            if not np.any(events & (self.cluster_ids_ == ordered[2])):
                raise ValueError("Highest-risk cluster has no documented default event; do not label it Default from centroid rank alone.")
        return self.map_labels(mapping)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = self._prepare(frame, fit=False)
        clusters = self.kmeans_.predict(x)
        distances = self.kmeans_.transform(x)
        out = frame.copy()
        out["cluster_id"] = clusters
        out["distance_to_centroid"] = distances[np.arange(len(out)), clusters]
        out["risk_label"] = [self.cluster_to_label_.get(int(k), f"Cluster_{int(k)}") for k in clusters]
        out["labels_validated"] = bool(self.cluster_to_label_)
        return out

    def cluster_summary(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        data = self.training_.copy() if frame is None else frame.copy()
        if "cluster_id" not in data:
            data["cluster_id"] = self.kmeans_.predict(self._prepare(data, fit=False))
        summary = data.groupby("cluster_id")[self.features].agg(["count", "mean", "median"])
        if self.cluster_to_label_:
            summary.insert(0, "risk_label", [self.cluster_to_label_[int(k)] for k in summary.index])
        return summary
