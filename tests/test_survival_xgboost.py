# third party
from pycox.datasets import metabric
from sklearn.model_selection import train_test_split

from organsync.survival_analysis.xgboost import XGBoostRiskEstimation


def test_survival_xgboost_plugin_fit_predict() -> None:
    test_plugin = XGBoostRiskEstimation()

    df = metabric.read_df()

    X = df.drop(["duration", "event"], axis=1)
    Y = df["event"]
    T = df["duration"]

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.1, random_state=0
    )

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    print(eval_time_horizons)
    y_pred = test_plugin.fit(
        X_train, T_train, Y_train, eval_times=eval_time_horizons
    ).predict(X_test, eval_time_horizons)

    assert y_pred.shape == (Y_test.shape[0], len(eval_time_horizons))
