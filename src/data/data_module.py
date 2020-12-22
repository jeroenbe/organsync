import pytorch_lightning as pl

import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

import pandas as pd

from src.data.utils import get_data_tuples

# NEXT STEP: incorporate make_dataset.py into this (def prepare_data(self))

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