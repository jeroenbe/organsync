import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from organsync.data.data_module import OrganDataModule

# OWN MODULES
from .utils import o_cols, x_cols

# NEXT STEP 1: incorporate make_dataset.py into this (def prepare_data(self))
# NEXT STEP 2: UNOSDataModule and UKRegDataModule share a lot of code -> make abstract OrganDataModule


class UNOSDataModule(OrganDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        is_synth: bool = False,
        test_size: float = 0.05,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            is_synth=is_synth,
            test_size=test_size,
        )

        self.data_dir = data_dir
        self.dims = (0, 141)

    def prepare_data(self) -> None:
        # Divide data in tuples as described in the paper.

        # LOAD
        liver_train = pd.read_csv(f"{self.data_dir}/liver_processed_train.csv")
        liver_test = pd.read_csv(f"{self.data_dir}/liver_processed_test.csv")

        liver_train = liver_train[liver_train.PSTATUS == 1]
        liver_test = liver_test[liver_test.PSTATUS == 1]

        # ONLY USE PRESENT COLS
        self.x_cols = np.intersect1d(liver_train.columns.values, x_cols)
        self.o_cols = np.intersect1d(liver_train.columns.values, o_cols)

        liver_train["Y"] = liver_train.PTIME
        liver_test["Y"] = liver_test.PTIME
        liver_train["CENS"] = liver_train.PSTATUS - 1
        liver_test["CENS"] = liver_test.PSTATUS - 1

        self.mean = liver_train.Y.mean()
        self.std = liver_train.Y.std()

        liver_train.Y -= self.mean
        liver_train.Y /= self.std

        liver_test.Y -= self.mean
        liver_test.Y /= self.std

        self._train_processed, self._test_processed = liver_train, liver_test


class UNOS2UKRegDataModule(OrganDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        replace_organ: int = -1,
        is_synth: bool = False,
        control: bool = False,
        test_size: float = 0.05,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            replace_organ=replace_organ,
            is_synth=is_synth,
            test_size=test_size,
        )

        self.data_dir = data_dir
        self.control = control
        self.dims = (0, 51)

    def prepare_data(self) -> None:
        DATA = pd.read_csv(f"{self.data_dir}/liver_processed.csv")
        self.scaler = joblib.load(f"{self.data_dir}/scaler")
        self.real_cols = np.load(
            f"{self.data_dir}/liver_processed_conts.npy", allow_pickle=True
        )

        self.x_cols = np.load(f"{self.data_dir}/x_cols.npy", allow_pickle=True)
        self.o_cols = np.load(f"{self.data_dir}/o_cols.npy", allow_pickle=True)

        # liver_train = pd.read_csv(f'{self.data_dir}/liver_processed_train.csv')
        # liver_test = pd.read_csv(f'{self.data_dir}/liver_processed_test.csv')

        if self.control:
            # liver_train = liver_train[(not liver_train.RECEIVED_TX)]
            # liver_test = liver_test[(not liver_test.RECEIVED_TX)]
            self.o_cols = []
            DATA = DATA[DATA.RECEIVED_TX is False]
        else:
            # liver_train = liver_train[liver_train.RECEIVED_TX]
            # liver_test = liver_test[liver_test.RECEIVED_TX]
            DATA = DATA[DATA.RECEIVED_TX]

        self.DATA = DATA
        liver_train, liver_test = train_test_split(DATA, test_size=self.test_size)

        self.mean = self.scaler.mean_[-1]
        self.std = self.scaler.scale_[-1]

        self._train_processed, self._test_processed = liver_train, liver_test


class UKRegDataModule(OrganDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        replace_organ: int = -1,
        is_synth: bool = False,
        test_size: float = 0.2,
        control: bool = False,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            replace_organ=replace_organ,
            is_synth=is_synth,
            test_size=test_size,
        )

        self.data_dir = data_dir
        self.dims = (0, 79)
        self.control = control

    def prepare_data(self) -> None:

        # AS REPORTED IN LAG-Document
        # TODO: provide link for this
        # TODO: remove leaked columns

        self.DATA = pd.read_csv(f"{self.data_dir}/data_preprocessed.csv", index_col=0)
        xm1 = np.load(f"{self.data_dir}/x_cols_m1.npy", allow_pickle=True)
        xm2 = np.load(f"{self.data_dir}/x_cols_m2.npy", allow_pickle=True)

        self.x_cols = np.union1d(xm1, xm2)
        self.x_cols = np.setdiff1d(self.x_cols, ["PSURV", "rwtime"])
        self.o_cols = np.load(f"{self.data_dir}/o_cols_m2.npy", allow_pickle=True)
        self.real_cols = np.load(f"{self.data_dir}/impute.npy", allow_pickle=True)
        self.real_cols = np.array([*self.real_cols, "Y"])

        self.scaler = joblib.load(f"{self.data_dir}/scaler")

        if self.control:
            self.o_cols = []
            self.DATA = self.DATA.loc[self.DATA.DCOD_0.isnull(), :]
        else:
            self.DATA.loc[self.DATA.DCOD_0.isnull(), self.o_cols] = self.replace_organ
        self.DATA.replace(np.nan, self.replace_organ, inplace=True)

        self._train_processed, self._test_processed = train_test_split(
            self.DATA, test_size=self.test_size
        )

        self.max = self._train_processed.Y.max()
        self.min = self._train_processed.Y.min()

        self.mean = self.scaler.mean_[-1]
        self.std = self.scaler.scale_[-1]
