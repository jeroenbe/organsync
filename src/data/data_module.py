import joblib

import pandas as pd
import numpy as np

import pytorch_lightning as pl

import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split


# OWN MODULES
from src.data.utils import get_data_tuples

# NEXT STEP 1: incorporate make_dataset.py into this (def prepare_data(self))
# NEXT STEP 2: UNOSDataModule and UKRegDataModule share a lot of code -> make abstract OrganDataModule

class UNOSDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

        self.dims = (0, 141)

        self.scale_constant = 4


    def setup(self, stage=None):
        # Data should be of form: {(X_i, O_ij, Y_i, delta_i)}

        X_train, O_train, Y_train, del_train, X_test, O_test, Y_test, del_test = get_data_tuples(self.data_dir)
        X_train = X_train[del_train.PSTATUS == 0]
        O_train = O_train[del_train.PSTATUS == 0]
        Y_train = Y_train[del_train.PSTATUS == 0]
        del_train = del_train[del_train.PSTATUS == 0]

        X_test = X_test[del_test.PSTATUS == 0]
        O_test = O_test[del_test.PSTATUS == 0]
        Y_test = Y_test[del_test.PSTATUS == 0]
        del_test = del_test[del_test.PSTATUS == 0]


        self.max = Y_train.max()
        self.min = Y_train.min()

        self.mean = Y_train.mean().to_numpy()
        self.std = Y_train.std().to_numpy()

        Y_train -= self.mean
        Y_train /= self.std

        Y_test -= self.mean
        Y_test /= self.std

        #Y_train -= self.min 
        #Y_train /= ((self.max - self.min) / self.scale_constant)

        #Y_test -= self.min
        #Y_test /= ((self.max - self.min)/ self.scale_constant)

        if stage == 'fit' or stage is None:
            X = torch.tensor(X_train.to_numpy(), dtype=torch.double)
            O = torch.tensor(O_train.to_numpy(), dtype=torch.double)
            Y = torch.tensor(Y_train.to_numpy(), dtype=torch.double)
            delt = torch.tensor(del_train.to_numpy(), dtype=torch.double)

            ds = TensorDataset(X, O, Y, delt)

            train_size = int(len(ds)*.8)
            self.train, self.val = random_split(ds, [train_size, len(ds) - train_size])

            self.dims = (X.size(0), X.size(1) + O.size(1))

        if stage == 'test' or stage is None:
            X = torch.tensor(X_test.to_numpy(), dtype=torch.double)
            O = torch.tensor(O_test.to_numpy(), dtype=torch.double)
            Y = torch.tensor(Y_test.to_numpy(), dtype=torch.double)
            delt = torch.tensor(del_test.to_numpy(), dtype=torch.double)

            self.test = TensorDataset(X, O, Y, delt)
            self.dims = (X.size(0), X.size(1) + O.size(1))
    

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class UKRegDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, replace_organ: int=-1):
        super().__init__()

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.replace_organ = replace_organ

        self.dims = (0, 81)

    def prepare_data(self):

        # AS REPORTED IN LAG-Document TODO: provide link for this
        self.DATA = pd.read_csv(f'{self.data_dir}/data_preprocessed.csv', index_col=0)
        self.xm1 = np.load(f'{self.data_dir}/x_cols_m1.npy', allow_pickle=True)
        self.xm2 = np.load(f'{self.data_dir}/x_cols_m2.npy', allow_pickle=True)
        self.om2 = np.load(f'{self.data_dir}/o_cols_m2.npy', allow_pickle=True)
        self.real_cols = np.load(f'{self.data_dir}/impute.npy', allow_pickle=True)
        self.scaler = joblib.load(f'{self.data_dir}/scaler')

        self.DATA.loc[self.DATA.DCOD_0.isnull(), self.om2] = self.replace_organ
        self.DATA.replace(np.nan, self.replace_organ, inplace=True)

        self._train_processed, self._test_processed = train_test_split(self.DATA, test_size=.2)

    
    def setup(self, stage=None):

        X_train, O_train, Y_train, del_train = (self._train_processed[np.union1d(self.xm1, self.xm2)],
                                                self._train_processed[self.om2],
                                                self._train_processed.Y,
                                                self._train_processed.CENS)
        X_test, O_test, Y_test, del_test = (self._test_processed[np.union1d(self.xm1, self.xm2)],
                                            self._test_processed[self.om2],
                                            self._test_processed.Y,
                                            self._test_processed.CENS)

        self.max = Y_train.max()
        self.min = Y_train.min()

        self.mean = self.scaler.mean_[-1]
        self.std = self.scaler.scale_[-1]
        
        if stage == 'fit' or stage is None:
            X = torch.tensor(X_train.to_numpy(), dtype=torch.double)
            O = torch.tensor(O_train.to_numpy(), dtype=torch.double)
            Y = torch.tensor(Y_train.to_numpy(), dtype=torch.double).view(-1, 1)
            delt = torch.tensor(del_train.to_numpy(), dtype=torch.double)

            ds = TensorDataset(X, O, Y, delt)

            train_size = int(len(ds)*.8)
            self.train, self.val = random_split(ds, [train_size, len(ds) - train_size])

            self.dims = (X.size(0), X.size(1) + O.size(1))


        if stage == 'test' or stage is None:
            X = torch.tensor(X_test.to_numpy(), dtype=torch.double)
            O = torch.tensor(O_test.to_numpy(), dtype=torch.double)
            Y = torch.tensor(Y_test.to_numpy(), dtype=torch.double).view(-1, 1)
            delt = torch.tensor(del_test.to_numpy(), dtype=torch.double)

            self.test = TensorDataset(X, O, Y, delt)
            self.dims = (X.size(0), X.size(1) + O.size(1))

    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
