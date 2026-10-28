"""OLS common-component estimation with an explicit identification check."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CommonComponentModel:
    """Fit rho = common + Z beta + residual; common is not automatically risk-free."""
    condition_number_limit: float = 1e10

    def fit(self, frame: pd.DataFrame, *, target: str, features: list[str]) -> "CommonComponentModel":
        required = [target, *features]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if frame.empty:
            raise ValueError("Cannot fit an empty frame.")
        data = frame[required].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < max(3, len(features) + 2):
            raise ValueError("Insufficient complete observations.")
        y = data[target].to_numpy(dtype=float)
        z = data[features].to_numpy(dtype=float) if features else np.empty((len(data), 0))
        if z.shape[1]:
            if np.linalg.matrix_rank(z) < z.shape[1]:
                raise ValueError("Covariate matrix is rank deficient.")
            cond = float(np.linalg.cond(z))
            if not np.isfinite(cond) or cond > self.condition_number_limit:
                raise ValueError("Covariate matrix is ill-conditioned.")
            projection = z @ np.linalg.solve(z.T @ z, z.T)
            my = y - projection @ y
            mone = np.ones(len(data)) - projection @ np.ones(len(data))
        else:
            my, mone = y.copy(), np.ones(len(data))
        denominator = float(mone @ mone)
        if denominator <= np.finfo(float).eps * max(1.0, float(np.linalg.norm(mone) ** 2)):
            raise ValueError("Common component is not separately identified from supplied features.")
        common = float((mone @ my) / denominator)
        beta = np.linalg.lstsq(z, y - common, rcond=None)[0] if z.shape[1] else np.empty(0)
        fitted = common + (z @ beta if z.shape[1] else 0.0)
        result = data.copy()
        result["common_component"] = common
        result["structural_fitted"] = fitted
        result["structural_residual"] = y - fitted
        self.target_, self.features_ = target, list(features)
        self.common_component_, self.coefficients_ = common, beta
        self.results_ = result
        self.residuals_ = pd.Series(y - fitted, index=data.index, name="structural_residual")
        self.n_observations_ = len(data)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "common_component_"):
            raise RuntimeError("Call fit before transform.")
        missing = [c for c in self.features_ if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        zdf = frame[self.features_].replace([np.inf, -np.inf], np.nan)
        if zdf.isna().any().any():
            raise ValueError("Features contain missing or non-finite values.")
        z = zdf.to_numpy(dtype=float) if self.features_ else np.empty((len(frame), 0))
        fitted = self.common_component_ + (z @ self.coefficients_ if z.shape[1] else 0.0)
        out = frame.copy()
        out["common_component"] = self.common_component_
        out["structural_fitted"] = fitted
        if self.target_ in frame:
            out["structural_residual"] = frame[self.target_].to_numpy(dtype=float) - fitted
        return out
