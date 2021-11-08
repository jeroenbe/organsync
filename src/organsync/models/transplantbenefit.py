from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from lifelines import CoxPHFitter


class UKELDModel:
    def __init__(
        self,
        data: pd.DataFrame,
        cols: list,
        duration_col: str,
        censor_col: str,
        penalizer: float = 0.1,
    ) -> None:
        self.data = data
        self.cols = cols
        self.duration_col = duration_col
        self.censor_col = censor_col
        self.penalizer = penalizer

        self.cph = CoxPHFitter(penalizer=self.penalizer)

    def load_cph(self, location: Path) -> None:
        self.cph = load(location)

    def save_cph(self, location: Path, name: str) -> None:
        location = Path(location)
        if not location.exists():
            location.mkdir()
        dump(self.cph, location / name)

    def fit(self) -> None:
        self.cph.fit(
            self.data.loc[:, [*self.cols, self.duration_col, self.censor_col]],
            duration_col=self.duration_col,
            event_col=self.censor_col,
        )

    def estimate(self, x: np.ndarray) -> np.ndarray:
        return self.cph.predict_expectation(x)
