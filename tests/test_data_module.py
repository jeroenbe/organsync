import numpy as np
from sklearn.model_selection import train_test_split

from organsync.data.data_module import OrganDataModule


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

    def prepare_data(self) -> None:
        self.real_cols = ["a", "b"]
        self.x_cols = ["a"]
        self.o_cols = ["b"]
        self.DATA = np.zeros((100, 2))
        self.mean = 1
        self.std = 0.5

        self._train_processed, self._test_processed = train_test_split(
            self.DATA, test_size=0.05
        )


def test_data_module_sanity() -> None:
    mock = MockDataModule(batch_size=16, replace_organ=1, is_synth=True)

    assert mock.batch_size == 16
    assert mock.replace_organ == 1
    assert mock.is_synth is True
    assert mock.test_size == 0.05
