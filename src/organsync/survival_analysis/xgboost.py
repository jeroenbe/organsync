# stdlib
from typing import Any, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgbse import XGBSEDebiasedBCE, XGBSEStackedWeibull
from xgbse.converters import convert_to_structured


class XGBoostRiskEstimation:
    booster = ["gbtree", "gblinear", "dart"]

    def __init__(
        self,
        colsample_bynode: float = 0.5,
        max_depth: int = 4,
        subsample: float = 0.5,
        learning_rate: float = 5e-2,
        min_child_weight: int = 1,
        tree_method: str = "hist",
        booster: int = 2,
        random_state: int = 0,
        objective: str = "cox",  # "aft", "cox"
        strategy: str = "weibull",  # "weibull", "debiased_bce"
        **kwargs: Any,
    ) -> None:
        surv_params = {}
        if objective == "aft":
            surv_params = {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.0,
            }
        else:
            surv_params = {
                "objective": "survival:cox",
                "eval_metric": "cox-nloglik",
            }
        xgboost_params = {
            # survival
            **surv_params,
            **kwargs,
            # basic xgboost
            "colsample_bynode": colsample_bynode,
            "max_depth": max_depth,
            "subsample": subsample,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "verbosity": 0,
            "tree_method": tree_method,
            "booster": XGBoostRiskEstimation.booster[booster],
            "random_state": random_state,
            "n_jobs": 2,
        }
        lr_params = {
            "C": 1e-3,
            "max_iter": 10000,
        }
        if strategy == "debiased_bce":
            base_model = XGBSEDebiasedBCE(xgboost_params, lr_params)
        elif strategy == "weibull":
            base_model = XGBSEStackedWeibull(xgboost_params)
        else:
            raise ValueError(f"unknown strategy {strategy}")

        self.model = base_model

    def fit(
        self,
        X: pd.DataFrame,
        T: pd.DataFrame,
        E: pd.DataFrame,
        eval_times: Optional[list] = None,
    ) -> "XGBoostRiskEstimation":
        y = convert_to_structured(T, E)

        (X_train, X_valid, y_train, y_valid) = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(
            X_train,
            y_train,
            num_boost_round=1500,
            validation_data=(X_valid, y_valid),
            early_stopping_rounds=10,
            time_bins=eval_times,
        )

        return self

    def _find_nearest(self, array: np.ndarray, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def predict(
        self,
        X: pd.DataFrame,
        time_horizons: list,
    ) -> pd.DataFrame:
        if len(time_horizons) < 1:
            raise ValueError("Invalid input for time horizons.")

        # surv, upper_ci, lower_ci = self.model.predict(X, return_ci = True)
        surv = self.model.predict(X)
        surv = surv.loc[:, ~surv.columns.duplicated()]

        preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])

        time_bins = surv.columns
        for t, eval_time in enumerate(time_horizons):
            nearest = self._find_nearest(time_bins, eval_time)
            preds_[:, t] = np.asarray(1 - surv[nearest])

        return pd.DataFrame(preds_, columns=time_horizons)
