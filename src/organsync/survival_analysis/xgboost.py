# stdlib
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
        n_estimators: int = 3,
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
        self.models = []
        for _ in range(n_estimators):
            if strategy == "debiased_bce":
                base_model = XGBSEDebiasedBCE(xgboost_params, lr_params)
            elif strategy == "weibull":
                base_model = XGBSEStackedWeibull(xgboost_params)
            else:
                raise ValueError(f"unknown strategy {strategy}")
            self.models.append(base_model)

    def get_time_bins(self, T, E, size=12):
        """
        Method to automatically define time bins
        """

        lower_bound = max(T[E == 0].min(), T[E == 1].min()) + 1
        upper_bound = min(T[E == 0].max(), T[E == 1].max()) - 1

        return np.linspace(lower_bound, upper_bound, size, dtype=int)

    def fit(
        self,
        X: pd.DataFrame,
        T: pd.DataFrame,
        E: pd.DataFrame,
        eval_times: Optional[List] = None,
    ) -> "XGBoostRiskEstimation":
        skf = StratifiedKFold(n_splits=len(self.models), shuffle=True, random_state=0)

        if eval_times:
            eval_times = self.get_time_bins(T, E)
            eval_times = list(sorted(np.unique(eval_times)))

        cv_idx = 0
        for train_index, test_index in skf.split(X, E):
            X_train = X.loc[X.index[train_index]]
            E_train = E.loc[E.index[train_index]]
            T_train = T.loc[T.index[train_index]]
            X_valid = X.loc[X.index[test_index]]
            E_valid = E.loc[E.index[test_index]]
            T_valid = T.loc[T.index[test_index]]

            y_train = convert_to_structured(T_train, E_train)
            y_valid = convert_to_structured(T_valid, E_valid)
            self.models[cv_idx].fit(
                X_train,
                y_train,
                num_boost_round=1500,
                validation_data=(X_valid, y_valid),
                early_stopping_rounds=10,
                time_bins=eval_times,
            )
            cv_idx += 1

        return self

    def _find_nearest(self, array: np.ndarray, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _generate_score(self, results: np.ndarray) -> Tuple[float, float]:
        percentile_val = 1.96
        return (
            np.mean(results, axis=0),
            percentile_val * np.std(results, axis=0) / np.sqrt(len(results)),
        )

    def predict(
        self,
        X: pd.DataFrame,
        time_horizons: list,
        return_ci: bool = False,
    ) -> pd.DataFrame:
        if len(time_horizons) < 1:
            raise ValueError("Invalid input for time horizons.")

        results = []
        for model in self.models:
            surv = model.predict(X)

            surv = surv.loc[:, ~surv.columns.duplicated()]

            preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])

            time_bins = surv.columns
            for t, eval_time in enumerate(time_horizons):
                nearest = self._find_nearest(time_bins, eval_time)
                preds_[:, t] = np.asarray(1 - surv[nearest])

            results.append(preds_)
        results = np.asarray(results)
        mean, std = self._generate_score(results)

        if return_ci:
            return (
                pd.DataFrame(mean, columns=time_horizons),
                pd.DataFrame(std, columns=time_horizons),
            )
        return pd.DataFrame(mean, columns=time_horizons)
