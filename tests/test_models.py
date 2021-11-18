import numpy as np
import pandas as pd

from organsync.models.confidentmatch import ConfidentMatch
from organsync.models.linear import MELD, MELD_na
from organsync.models.organite_network import OrganITE_Network
from organsync.models.organsync_network import OrganSync_Network
from organsync.models.transplantbenefit import UKELDModel


def test_meld_sanity() -> None:
    assert MELD().score(0.7, 1.0, 0.7) == 1.6648227489785334
    assert MELD_na().score(0.7, 1.0, 0.7, 100) == 40.0


def test_confidentmatch_sanity() -> None:
    dummy = pd.DataFrame(np.zeros((100, 3)), columns=["x", "o", "y"])
    mock = ConfidentMatch(3, dummy, x_col=["x"], o_col=["o"], y_col="y", H={})

    assert mock.k == 3
    assert mock.x_col == ["x"]
    assert mock.o_col == ["o"]
    assert mock.y_col == "y"


def test_ukeld_sanity() -> None:
    dummy = pd.DataFrame(np.zeros((100, 3)), columns=["x", "o", "y"])
    mock = UKELDModel(dummy, cols=["x", "o", "y"], duration_col="o", censor_col="y")

    assert mock.cols == ["x", "o", "y"]
    assert mock.penalizer == 0.1
    assert mock.censor_col == "y"
    assert mock.duration_col == "o"
    assert mock.cph is not None


def test_organite_sanity() -> None:
    mock = OrganITE_Network(
        input_dim=2,
        hidden_dim=3,
        output_dim=4,
        lr=0.05,
        gamma=0.06,
        lambd=0.07,
        kappa=0.08,
        weight_decay=0.09,
        n_clusters=10,
        num_hidden_layers=11,
    )

    assert mock.lr == 0.05
    assert mock.gamma == 0.06
    assert mock.lambd == 0.07
    assert mock.kappa == 0.08
    assert mock.weight_decay == 0.09
    assert mock.n_clusters == 10
    assert len(mock.representation) == 37
    assert len(mock.propensity) == 5
    assert len(mock.output) == 3


def test_organsync_sanity() -> None:
    mock = OrganSync_Network(
        input_dim=2,
        hidden_dim=3,
        output_dim=4,
        lr=0.05,
        gamma=0.06,
        lambd=0.07,
        weight_decay=0.09,
        num_hidden_layers=11,
    )

    assert mock.lr == 0.05
    assert mock.gamma == 0.06
    assert mock.lambd == 0.07
    assert mock.weight_decay == 0.09
    assert len(mock.representation) == 39
