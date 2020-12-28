import joblib

import pandas as pd
import numpy as np

import pytorch_lightning as pl

import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from abc import abstractclassmethod


# OWN MODULES
from src.data.utils import x_cols, o_cols, UNOS_2_UKReg_mapping

# NEXT STEP 1: incorporate make_dataset.py into this (def prepare_data(self))
# NEXT STEP 2: UNOSDataModule and UKRegDataModule share a lot of code -> make abstract OrganDataModule


class OrganDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size:int, replace_organ: int=-1, is_synth: bool=False):
        super().__init__()

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.is_synth = is_synth

        self.replace_organ = replace_organ

    @abstractclassmethod
    def prepare_data(self):
        pass
    
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
    
    def setup(self, stage=None):
        X_train, O_train, Y_train, del_train = (self._train_processed[self.x_cols],
                                                self._train_processed[self.o_cols],
                                                self._train_processed.Y,
                                                self._train_processed.CENS)
        X_test, O_test, Y_test, del_test = (self._test_processed[self.x_cols],
                                            self._test_processed[self.o_cols],
                                            self._test_processed.Y,
                                            self._test_processed.CENS)

        if self.is_synth:
            # 1. create Y_train
            # 2. suffle X and O in test (no bias)
            # 3. create Y_test
            pass
        
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



class UNOSDataModule(OrganDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__(data_dir=data_dir, batch_size=batch_size)

        self.dims = (0, 141)


    def prepare_data(self):
        # Divide data in tuples as described in the paper.

        # LOAD
        liver_train = pd.read_csv(f'{self.data_dir}/liver_processed_train.csv')
        liver_test = pd.read_csv(f'{self.data_dir}/liver_processed_test.csv')

        liver_train = liver_train[liver_train.PSTATUS == 1]
        liver_test = liver_test[liver_test.PSTATUS == 1]

        # ONLY USE PRESENT COLS
        self.x_cols = np.intersect1d(liver_train.columns.values, x_cols)
        self.o_cols = np.intersect1d(liver_train.columns.values, o_cols)

        liver_train['Y'] = liver_train.PTIME
        liver_test['Y'] = liver_test.PTIME
        liver_train['CENS'] = liver_train.PSTATUS-1
        liver_test['CENS'] = liver_test.PSTATUS-1

        self.mean = liver_train.Y.mean()
        self.std = liver_train.Y.std()

        liver_train.Y -= self.mean
        liver_train.Y /= self.std

        liver_test.Y -= self.mean
        liver_test.Y /= self.std

        self._train_processed, self._test_processed = liver_train, liver_test

class UNOS2UKRegDataModule(OrganDataModule):
    def __init__(self, data_dir: str, batch_size: int, replace_organ: int=-1):
        super().__init__(data_dir=data_dir, batch_size=batch_size, replace_organ=replace_organ)

        self.dims = (0, 55)

    def prepare_data(self):
        liver_train = pd.read_csv(f'{self.data_dir}/liver_processed_train.csv')
        liver_test = pd.read_csv(f'{self.data_dir}/liver_processed_test.csv')

        self.scaler = joblib.load(f'{self.data_dir}/scaler')

        U2U_mapped = np.array(list(UNOS_2_UKReg_mapping.keys()))
        x_cols_present = np.intersect1d(U2U_mapped, x_cols)
        o_cols_present = np.intersect1d(U2U_mapped, o_cols)

        self.x_cols = np.array([UNOS_2_UKReg_mapping[k] for k in x_cols_present])
        self.o_cols = np.array([UNOS_2_UKReg_mapping[k] for k in o_cols_present])

        # TEMPORARY SOLUTION
        self.x_cols = ['RAGE', 'RCREAT', 'RINR', 'RSODIUM', 'RALBUMIN',
            'SERUM_BILIRUBIN', 'INR', 'SERUM_CREATININE', 'SERUM_SODIUM',
            'regyr', 'RBILIRUBIN', 'PRIMARY_LIVER_DISEASE',  
            'CENS', 'Y', 'BILIR_SOD', 'BILIR_DG',
            'RASCITES_0', 'RASCITES_1', 'RASCITES_2', 'RASCITES_3',
            'RENAL_SUPPORT_-1', 'RENAL_SUPPORT_0', 'RENAL_SUPPORT_1', 'SEX_0',
            'SEX_1', 'RHCV_-1', 'RHCV_0', 'RHCV_1', 'RHCV_2', 'RHCV_3',
            'RENCEPH_-1', 'RENCEPH_0', 'RENCEPH_1', 'RENCEPH_2', 'RENCEPH_3',
            'PATIENT_LOCATION_-1', 'PATIENT_LOCATION_0', 'PATIENT_LOCATION_1',
            'PATIENT_LOCATION_2', 'RAB_SURGERY_-1', 'RAB_SURGERY_0',
            'RAB_SURGERY_1', 'RAB_SURGERY_2'
        ]

        self.o_cols = [
            'DAGE', 'DBMI', 'DGRP_-1', 'DGRP_0', 'DGRP_1', 
            'DGRP_DG', 'DGRP_AGE', 'DGRP_RCREA', 'DGRP_RABS',
             'AGE_CREAT', 'HCV_AGE','AGE_DG', 
        ]

        self.mean = self.scaler.mean_[-1]
        self.std = self.scaler.scale_[-1]

        self._train_processed, self._test_processed = liver_train, liver_test



class UKRegDataModule(OrganDataModule):
    def __init__(self, data_dir: str, batch_size: int, replace_organ: int=-1):
        super().__init__(data_dir=data_dir, batch_size=batch_size, replace_organ=replace_organ)

        self.dims = (0, 81)

    def prepare_data(self):

        # AS REPORTED IN LAG-Document 
        # TODO: provide link for this
        # TODO: split train-test before processing
        self.DATA = pd.read_csv(f'{self.data_dir}/data_preprocessed.csv', index_col=0)
        xm1 = np.load(f'{self.data_dir}/x_cols_m1.npy', allow_pickle=True)
        xm2 = np.load(f'{self.data_dir}/x_cols_m2.npy', allow_pickle=True)
        self.x_cols = np.union1d(xm1, xm2)
        self.o_cols = np.load(f'{self.data_dir}/o_cols_m2.npy', allow_pickle=True)
        #self.real_cols = np.load(f'{self.data_dir}/impute.npy', allow_pickle=True)
        self.scaler = joblib.load(f'{self.data_dir}/scaler')

        self.DATA.loc[self.DATA.DCOD_0.isnull(), self.o_cols] = self.replace_organ
        self.DATA.replace(np.nan, self.replace_organ, inplace=True)

        self._train_processed, self._test_processed = train_test_split(self.DATA, test_size=.2)


        self.max = self._train_processed.Y.max()
        self.min = self._train_processed.Y.min()

        self.mean = self.scaler.mean_[-1]
        self.std = self.scaler.scale_[-1]
