from abc import abstractclassmethod
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split


class OrganDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        replace_organ: int = -1,
        is_synth: bool = False,
        test_size: float = 0.05,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.is_synth = is_synth
        self.test_size = test_size

        self.replace_organ = replace_organ

    @abstractclassmethod
    def prepare_data(self) -> None:
        ...

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=joblib.cpu_count()
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=joblib.cpu_count()
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=joblib.cpu_count()
        )

    def setup(self, stage: Optional[str] = None) -> None:
        X_train, O_train, Y_train, del_train = (
            self._train_processed[self.x_cols],
            self._train_processed[self.o_cols],
            self._train_processed.Y,
            self._train_processed.CENS,
        )
        X_test, O_test, Y_test, del_test = (
            self._test_processed[self.x_cols],
            self._test_processed[self.o_cols],
            self._test_processed.Y,
            self._test_processed.CENS,
        )

        if self.is_synth:
            self.theta_x = np.random.uniform(size=len(self.x_cols))
            self.theta_o = np.random.uniform(size=len(self.o_cols))

            prev_mean = Y_train.mean()
            prev_std = Y_train.std()

            prev_min = Y_train.min()
            prev_max = Y_train.max()

            # 1. create Y_train
            Y_train = np.exp((X_train @ self.theta_x + O_train @ self.theta_o))
            Y_train += np.random.normal(scale=0.1, size=Y_train.shape)
            synth_std = Y_train.std()
            synth_mean = Y_train.mean()

            Y_train *= prev_std / synth_std
            Y_train += prev_mean - synth_mean

            Y_train[Y_train < prev_min] = prev_min
            Y_train[Y_train > prev_max] = prev_max

            # 2. suffle X and O in test (no bias)
            O_test = O_test.sample(frac=1)

            # 3. create Y_test
            Y_test = np.exp((X_test @ self.theta_x + O_test @ self.theta_o))
            Y_test *= prev_std / synth_std
            Y_test += prev_mean - synth_mean

            Y_test[Y_test < prev_min] = prev_min
            Y_test[Y_test > prev_max] = prev_max

        if stage == "fit" or stage is None:
            X = torch.tensor(X_train.to_numpy(), dtype=torch.double)
            O = torch.tensor(O_train.to_numpy(), dtype=torch.double)
            Y = torch.tensor(Y_train.to_numpy(), dtype=torch.double).view(-1, 1)
            delt = torch.tensor(del_train.to_numpy(), dtype=torch.double)

            ds = TensorDataset(X, O, Y, delt)

            train_size = int(len(ds) * 0.8)
            self.train, self.val = random_split(ds, [train_size, len(ds) - train_size])

            self.dims = (X.size(0), X.size(1) + O.size(1))

        if stage == "test" or stage is None:

            X = torch.tensor(X_test.to_numpy(), dtype=torch.double)
            O = torch.tensor(O_test.to_numpy(), dtype=torch.double)
            Y = torch.tensor(Y_test.to_numpy(), dtype=torch.double).view(-1, 1)
            delt = torch.tensor(del_test.to_numpy(), dtype=torch.double)

            self.test = TensorDataset(X, O, Y, delt)
            self.dims = (X.size(0), X.size(1) + O.size(1))

        if stage is None:
            self.all = ConcatDataset([self.train, self.test])
            self._all_processed = pd.concat(
                [
                    self._test_processed.copy(deep=True),
                    self._train_processed.copy(deep=True),
                ]
            )
