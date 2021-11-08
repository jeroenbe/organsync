from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class ConfidentMatch:
    def __init__(
        self,
        k: int,
        data: pd.DataFrame,
        x_col: list,
        o_col: list,
        y_col: str,
        H: dict,
        test_size: float = 0.2,
    ) -> None:

        self.k = k

        self.data = data
        self.x_col = x_col
        self.o_col = o_col
        self.y_col = y_col

        X, y = self.data[[*self.x_col, *self.o_col]], self.data[y_col]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, y, test_size=test_size
        )

        # H is the hypothesis space. It is a list of
        # different predictors [X, O] -> Y.
        self.H = self._init_H_(H)

        self.d: dict = dict()

        self.kmeans = KMeans(n_clusters=self.k).fit(self.X_train)

    def _init_H_(self, H: dict) -> dict:
        estimators = dict()
        for k, v in H.items():
            estimators[k] = v[0](**v[1])
        return estimators

    def _train(self) -> None:
        groups = self.X_train.groupby(by="partition")
        groups_test = self.X_test.groupby(by="partition")

        for k, v in groups.indices.items():
            perf = dict()
            models = dict()

            X_train_v, Y_train_v = self.X_train.iloc[v], self.Y_train.iloc[v]
            if k in groups_test.indices:
                X_test_v, Y_test_v = (
                    self.X_test.iloc[groups_test.indices[k]],
                    self.Y_test.iloc[groups_test.indices[k]],
                )
            else:
                X_test_v, Y_test_v = self.X_train.iloc[v], self.Y_train.iloc[v]

            for A_name, A in self.H.items():
                models[A_name] = A.fit(
                    X_train_v.drop("partition", axis=1).to_numpy(), Y_train_v.to_numpy()
                )

                y_predicted = models[A_name].predict(X_test_v.drop("partition", axis=1))
                perf[A_name] = mean_squared_error(Y_test_v, y_predicted)

            best_A = min(perf, key=lambda k: perf[k])
            self.d[k] = (best_A, models[best_A])

    def _get_partitions(self) -> np.ndarray:
        self.X_train["partition"] = self.kmeans.predict(self.X_train)
        self.X_test["partition"] = self.kmeans.predict(self.X_test)

    def estimate(self, X: np.ndarray) -> float:
        partition = self.kmeans.predict(X.reshape(1, -1)).item()
        return self.d[partition][1].predict(X.reshape(1, -1)).item()

    def save(self, location: Path, online: bool = False) -> None:
        location = Path(location)

        if not location.exists() and not online:
            location.mkdir()
        dump(self.d, location / "d")
        dump(self.kmeans, location / "kmeans")

    def load(self, location: Path) -> None:
        location = Path(location)
        self.d = load(location / "d")
        self.kmeans = load(location / "kmeans")
