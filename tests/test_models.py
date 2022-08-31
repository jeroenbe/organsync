import numpy as np
import pandas as pd
import pytest

from organsync.models.confidentmatch import ConfidentMatch
from organsync.models.linear import MELD, MELD3, MELD_na
from organsync.models.organite_network import OrganITE_Network
from organsync.models.organsync_network import OrganSync_Network


def test_meld_sanity() -> None:
    assert MELD().score(
        serum_bilirubin=0.3, inr=0.8, serum_creatinine=0.7
    ) == pytest.approx(6, 0.5)
    assert MELD().score(
        serum_bilirubin=1.9, inr=1.2, serum_creatinine=1.3
    ) == pytest.approx(13, 0.5)


def test_meldna_sanity() -> None:
    assert MELD_na().score(
        serum_bilirubin=0.3, inr=0.8, serum_creatinine=0.7, serum_sodium=136
    ) == pytest.approx(7, 0.5)

    assert MELD_na().score(
        serum_bilirubin=1.9, inr=1.2, serum_creatinine=1.3, serum_sodium=115
    ) == pytest.approx(23, 0.1)


def test_meld3_sanity() -> None:
    score = MELD3().score(
        sex="M",
        serum_bilirubin=0.3,
        inr=0.8,
        serum_creatinine=0.7,
        serum_sodium=136,
        serum_albumin=2.5,
    )
    assert score == pytest.approx(7, 0.5)

    score = MELD3().score(
        sex="F",
        serum_bilirubin=0.3,
        inr=0.8,
        serum_creatinine=0.7,
        serum_sodium=136,
        serum_albumin=2.5,
    )
    assert score == pytest.approx(9.6, 0.1)

    score = MELD3().score(
        sex="M",
        serum_bilirubin=1.9,
        inr=1.2,
        serum_creatinine=1.3,
        serum_sodium=136,
        serum_albumin=3.6,
    )
    assert score == pytest.approx(14, 0.1)


def test_confidentmatch_sanity() -> None:
    dummy = pd.DataFrame(np.zeros((100, 3)), columns=["x", "o", "y"])
    mock = ConfidentMatch(3, dummy, x_col=["x"], o_col=["o"], y_col="y", H={})

    assert mock.k == 3
    assert mock.x_col == ["x"]
    assert mock.o_col == ["o"]
    assert mock.y_col == "y"


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
