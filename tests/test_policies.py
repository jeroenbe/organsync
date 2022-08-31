import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from organsync.data.data_module import OrganDataModule
from organsync.models.inference import (
    Inference_OrganITE,
    Inference_OrganITE_VAE,
    Inference_OrganSync,
)
from organsync.models.organite_network import OrganITE_Network, OrganITE_Network_VAE
from organsync.models.organsync_network import OrganSync_Network
from organsync.policies import MELD, MELD3, MELD_na, OrganITE, OrganSync


class MockScaler:
    def inverse_transform(self, d: pd.DataFrame) -> pd.DataFrame:
        return d


class MockDataModule(OrganDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        replace_organ: int = -1,
        is_synth: bool = False,
        test_size: float = 0.05,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            replace_organ=replace_organ,
            is_synth=is_synth,
            test_size=test_size,
        )
        self.prepare_data()

    def prepare_data(self) -> None:
        self.real_cols = [
            "SEX",
            "SERUM_BILIRUBIN",
            "INR",
            "SERUM_CREATININE",
            "SERUM_SODIUM",
            "SERUM_ALBUMIN",
            "Y",
            "CENS",
        ]
        self.x_cols = self.real_cols
        self.o_cols = ["Y"]
        self.DATA = pd.DataFrame(
            np.zeros((10000, len(self.real_cols))), columns=self.real_cols
        )
        self.DATA["SEX"] = np.random.choice([0, 1], len(self.DATA))

        self.mean = 1
        self.std = 0.5
        self.scaler = MockScaler()

        self._train_processed, self._test_processed = train_test_split(
            self.DATA, test_size=0.05
        )


def test_meld_policy() -> None:
    mock_data = MockDataModule()
    samples = mock_data._test_processed.sample(5)

    mock = MELD("meld", samples.index.tolist(), mock_data)

    assert mock.name == "meld"
    assert len(mock.waitlist) == 5
    assert mock.dm == mock_data


def test_meld_na_policy() -> None:
    mock_data = MockDataModule()
    samples = mock_data._test_processed.sample(5)

    mock = MELD_na("meld_na", samples.index, mock_data)

    assert mock.name == "meld_na"
    assert len(mock.waitlist) == 5
    assert mock.dm == mock_data


def test_meld3_policy() -> None:
    mock_data = MockDataModule()
    samples = mock_data._test_processed.sample(5)

    mock = MELD3("meld3", samples.index, mock_data)

    assert mock.name == "meld3"
    assert len(mock.waitlist) == 5
    assert mock.dm == mock_data


def test_organite_policy() -> None:
    mock_data = MockDataModule()

    mock_inference = Inference_OrganITE(
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
    mock_inference_vae = Inference_OrganITE_VAE(
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

    mock = OrganITE(
        "organite",
        [],
        mock_data,
        inference_ITE=mock_inference,
        inference_VAE=mock_inference_vae,
    )

    assert mock.name == "organite"
    assert len(mock.waitlist) == 0
    assert mock.dm == mock_data


def test_organsync_policy() -> None:
    mock_data = MockDataModule()
    mock_data.prepare_data()
    mock_data.setup(stage="fit")

    mock_inference = Inference_OrganSync(
        OrganSync_Network(
            input_dim=mock_data.dims[1],
            hidden_dim=3,
            output_dim=4,
            lr=0.05,
            gamma=0.06,
            lambd=0.07,
            weight_decay=0.09,
            num_hidden_layers=11,
        ).double(),
        1,
        2,
    )

    mock = OrganSync(
        "organsync",
        [],
        mock_data,
        K=2,
        inference_0=mock_inference,
        inference_1=mock_inference,
    )

    assert mock.name == "organsync"
    assert len(mock.waitlist) == 0
    assert mock.dm == mock_data
    assert mock.K == 2
