import os, click, wandb, torch

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from sklearn.metrics import mean_squared_error

from joblib import load, dump

# OWN MODULES
from src.data.data_module import UNOSDataModule, UNOS2UKRegDataModule, UKRegDataModule

class UKELDModel:

    def __init__(self, data: pd.DataFrame, cols: np.ndarray, duration_col: str, censor_col: str, penalizer: float=.0):
        self.data = data
        self.cols = cols
        self.duration_col = duration_col
        self.censor_col = censor_col
        self.penalizer=penalizer
        
        self.cph = CoxPHFitter(penalizer=self.penalizer)



    def load_cph(self, location: str):
        self.cph = load(location)


    def save_cph(self, location: str, name: str):
        if not os.path.exists(location):
            os.makedirs(location)
        dump(self.cph, f'{location}/{name}')

    def fit(self):
        self.cph.fit(self.data.loc[:, [*self.cols, self.duration_col, self.censor_col]], duration_col=self.duration_col, event_col=self.censor_col)

    def estimate(self, x):
        return self.cph.predict_expectation(x)


@click.command()
@click.option('--wb_run', type=str, default='organsync-tb-u2u')
@click.option('--run_amount', type=int, default=1)
@click.option('--group', type=str, default=None)
@click.option('--data', type=str, default='U2U')
@click.option('--data_dir', type=str, default='./data/processed_UNOS2UKReg_no_split')
@click.option('--control', type=click.BOOL, default=False)
@click.option('--is_synth', type=click.BOOL, default=False)
@click.option('--test_size', type=float, default=.05)
@click.option('--penalizer', type=float, default=.1)
def train(
        wb_run,
        run_amount,
        group,
        data,
        data_dir,
        control,
        is_synth,
        test_size,
        penalizer):

    for _ in range(run_amount):
        # MANUAL LOGGING
        run = wandb.init(project=wb_run, reinit=True, group=group)

        run.config.is_synth = is_synth
        run.config.test_size = test_size

        # LOAD DATA
        if data == 'UNOS':
            dm = UNOSDataModule(data_dir, batch_size=256, is_synth=is_synth, test_size=test_size)
        if data == 'U2U':
            dm = UNOS2UKRegDataModule(data_dir, batch_size=256, is_synth=is_synth, control=control, test_size=test_size)
        else:
            dm = UKRegDataModule(data_dir, batch_size=256, is_synth=is_synth, test_size=test_size)
        dm.prepare_data()
        dm.setup(stage='fit')

        # CONSTRUCT MODEL
        DATA = dm._train_processed
        DATA.CENS = np.abs(DATA.CENS - 1)

        ukeld = UKELDModel(data=DATA, cols=np.union1d(dm.x_cols, dm.o_cols), censor_col='CENS', duration_col='Y', penalizer=penalizer)

        
        # TRAIN MODEL
        ukeld.fit()

        # TEST NETWORK
        dm.setup(stage='test')
        X, O, Y, _ = dm.test_dataloader().dataset.tensors
        X = torch.cat((X, O), dim=1).numpy()
        Y = Y.numpy()
        
        Y_ = ukeld.estimate(X)
        Y_ = Y_.to_numpy().reshape(-1, 1)

        rmse = np.sqrt(mean_squared_error(Y_, Y))


        # scale
        Y_ = Y_ * dm.std + dm.mean
        Y = Y * dm.std + dm.mean

        diff = np.abs(Y_ - Y).mean()

        run.log({
            'test - RMSE': rmse,
            'test - mean diff in days': diff
        })

        # SAVING
        ukeld.save_cph(location=run.dir, name=f'{data}_cph')

    wandb.finish()

if __name__ == "__main__":
    train()


