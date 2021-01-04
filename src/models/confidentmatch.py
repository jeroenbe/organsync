import os, click

import wandb, torch

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor



from joblib import load, dump

# OWN MODULES
from src.data.data_module import UNOS2UKRegDataModule, UKRegDataModule, UNOSDataModule

class ConfidentMatch:

    def __init__(
            self,
            k: int,
            data: pd.DataFrame,
            x_col: np.ndarray,
            o_col: np.ndarray,
            y_col: str,
            H: dict,
            test_size: float=.2):

        self.k = k

        self.data = data
        self.x_col = x_col
        self.o_col = o_col
        self.y_col = y_col

        X, y = self.data[[*self.x_col, *self.o_col]], self.data[y_col]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, y, test_size=test_size)
        
        # H is the hypothesis space. It is a list of 
        # different predictors [X, O] -> Y.
        self.H = self._init_H_(H)

        self.d = dict()

        self.kmeans = KMeans(n_clusters=self.k).fit(self.X_train)

    def _init_H_(self, H):
        estimators = dict()
        for k, v in H.items():
            estimators[k] = v[0](**v[1])
        return estimators

    def _train(self):
        groups = self.X_train.groupby(by='partition')
        groups_test = self.X_test.groupby(by='partition')

        for k, v in groups.indices.items():
            perf = dict()
            models = dict()

            X_train_v, Y_train_v = self.X_train.iloc[v], self.Y_train.iloc[v]
            if k in groups_test.indices:
                X_test_v, Y_test_v = self.X_test.iloc[groups_test.indices[k]], self.Y_test.iloc[groups_test.indices[k]]
            else:
                X_test_v, Y_test_v = self.X_train.iloc[v], self.Y_train.iloc[v]
            

            for A_name, A in self.H.items():
                models[A_name] = A.fit(X_train_v.drop('partition', axis=1).to_numpy(), Y_train_v.to_numpy())

                y_predicted = models[A_name].predict(X_test_v.drop('partition', axis=1))
                perf[A_name] = mean_squared_error(Y_test_v, y_predicted)
            

            best_A = min(perf, key=perf.get)
            self.d[k] = (best_A, models[best_A])


    def _get_partitions(self) -> np.ndarray:
        self.X_train['partition'] = self.kmeans.predict(self.X_train)
        self.X_test['partition'] = self.kmeans.predict(self.X_test)

    def estimate(self, X: np.ndarray) -> float:
        partition = self.kmeans.predict(X.reshape(1, -1)).item()
        return self.d[partition][1].predict(X.reshape(1, -1)).item()

    def save(self, location: str, online: bool=False):
        if not os.path.exists(location) and not online:
            os.makedirs(location)
        dump(self.d, f'{location}/d')
        dump(self.kmeans, f'{location}/kmeans')

    def load(self, location: str):
        self.d = load(f'{location}/d')
        self.kmeans = load(f'{location}/kmeans')


@click.command()
@click.option('--n_clusters', type=int, default=15)
@click.option('--epochs', type=int, default=50)
@click.option('--wb_run', type=str, default='organsync-cm-u2u')
@click.option('--run_amount', type=int, default=1)
@click.option('--group', type=str, default=None)
@click.option('--data', type=str, default='U2U')
@click.option('--data_dir', type=str, default='./data/processed_UNOS2UKReg_no_split')
@click.option('--control', type=click.BOOL, default=False)
@click.option('--is_synth', type=click.BOOL, default=False)
@click.option('--test_size', type=float, default=.05)
def train(
        n_clusters,
        epochs,
        wb_run,
        run_amount,
        group,
        data,
        data_dir,
        control,
        is_synth,
        test_size):

    for _ in range(run_amount):
        # MANUAL LOGGING
        run = wandb.init(project=wb_run, reinit=True, group=group)

        run.config.n_clusters = n_clusters
        run.config.epochs = epochs
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

        cm_kwargs = {
            'k': n_clusters, 'x_col': dm.x_cols, 'y_col': 'Y', 'H': {
                'RFR': (RandomForestRegressor, {}),
                'SVR': (svm.SVR, {}),
                'MLPR': (MLPRegressor, {'hidden_layer_sizes': (30, 100, 100, 30), 'max_iter': epochs}),
            }
        }
        
        # CONSTRUCT MODEL
        cm = ConfidentMatch(data=dm._train_processed, o_col=dm.o_cols, **cm_kwargs)

        
        # TRAIN NETWORK
        cm._get_partitions()
        cm._train()

        # TEST NETWORK
        dm.setup(stage='test')
        X, O, Y, _ = dm.test_dataloader().dataset.tensors
        X = torch.cat((X, O), dim=1).numpy()
        Y = Y.numpy()
        
        Y_ = np.array([cm.estimate(x) for x in X])

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
        cm.save(location=run.dir)

    wandb.finish()

if __name__ == "__main__":
    train()
