import numpy as np
import pandas as pd
import pytest

from organsync.models.confidentmatch import ConfidentMatch
from organsync.models.inference import (
    Inference_ConfidentMatch,
    Inference_OrganITE,
    Inference_OrganITE_VAE,
    Inference_OrganSync,
)
from organsync.models.organite_network import OrganITE_Network, OrganITE_Network_VAE
from organsync.models.organsync_network import OrganSync_Network


def test_organsync_inference() -> None:
    mock = Inference_OrganSync(
        OrganSync_Network(
            input_dim=2,
            hidden_dim=3,
            output_dim=4,
            lr=0.05,
            gamma=0.06,
            lambd=0.07,
            weight_decay=0.09,
            num_hidden_layers=11,
        ),
        1,
        2,
    )

    assert isinstance(mock.model, OrganSync_Network)
    assert mock.mean == 1
    assert mock.std == 2

    with pytest.raises(AssertionError):
        Inference_OrganSync("fail", 1, 2)


def test_organite_inference() -> None:
    mock = Inference_OrganITE(
        OrganITE_Network(
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
        ),
        1,
        2,
    )

    assert isinstance(mock.model, OrganITE_Network)
    assert mock.mean == 1
    assert mock.std == 2

    with pytest.raises(AssertionError):
        Inference_OrganITE("fail", 1, 2)


def test_organite_vae_inference() -> None:
    mock = Inference_OrganITE_VAE(
        OrganITE_Network_VAE(
            input_dim=2,
            hidden_dim=3,
            output_dim=4,
            lr=0.05,
            gamma=0.06,
            weight_decay=0.09,
        ),
        1,
        2,
    )

    assert isinstance(mock.model, OrganITE_Network_VAE)
    assert mock.mean == 1
    assert mock.std == 2


def test_confident_match_inference() -> None:
    dummy = pd.DataFrame(np.zeros((100, 3)), columns=["x", "o", "y"])
    model = ConfidentMatch(3, dummy, x_col=["x"], o_col=["o"], y_col="y", H={})

    mock = Inference_ConfidentMatch(model, 1, 2)

    assert isinstance(mock.model, ConfidentMatch)
    assert mock.mean == 1
    assert mock.std == 2

    with pytest.raises(AssertionError):
        Inference_ConfidentMatch("fail", 1, 2)
