# stdlib
from typing import Any

import numpy as np
import pandas as pd

# third party
from lifelines import CoxPHFitter


class CoxPH:
    def __init__(self) -> None:
        self.model = CoxPHFitter(penalizer=0.01)

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CoxPH":
        if len(args) < 2:
            raise ValueError("Invalid input for fit. Expecting X, T and Y.")

        T = args[0]
        Y = args[1]

        X = X.reset_index(drop=True)
        T = T.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        df = pd.concat([X, T, Y], axis=1)
        df.columns = [x for x in X.columns] + ["time", "label"]

        self.model.fit(df, duration_col="time", event_col="label", **kwargs)

        return self

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if len(args) < 1:
            raise ValueError("Invalid input for predict. Expecting X and time horizon.")

        time_horizons = args[0]

        surv = self.model.predict_survival_function(X)
        surv_times = np.asarray(surv.index).astype(int)
        surv = np.asarray(surv.T)
        preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])

        for t, eval_time in enumerate(time_horizons):
            tmp_time = np.where(eval_time <= surv_times)[0]
            if len(tmp_time) == 0:
                preds_[:, t] = 1.0 - surv[:, 0]
            else:
                preds_[:, t] = 1.0 - surv[:, tmp_time[0]]

        return pd.DataFrame(preds_, columns=time_horizons)
