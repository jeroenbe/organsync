import click
import numpy as np
import torch
import wandb
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# OWN MODULES
from experiments.data.data_module import (
    UKRegDataModule,
    UNOS2UKRegDataModule,
    UNOSDataModule,
)
from organsync.models.confidentmatch import ConfidentMatch


@click.command()
@click.option("--n_clusters", type=int, default=15)
@click.option("--epochs", type=int, default=50)
@click.option("--wb_run", type=str, default="organsync-cm-u2u")
@click.option("--run_amount", type=int, default=1)
@click.option("--group", type=str, default=None)
@click.option("--data", type=str, default="U2U")
@click.option("--data_dir", type=str, default="./data/processed_UNOS2UKReg_no_split")
@click.option("--control", type=click.BOOL, default=False)
@click.option("--is_synth", type=click.BOOL, default=False)
@click.option("--test_size", type=float, default=0.05)
def train(
    n_clusters: int,
    epochs: int,
    wb_run: str,
    run_amount: int,
    group: str,
    data: str,
    data_dir: str,
    control: bool,
    is_synth: bool,
    test_size: float,
) -> None:

    for _ in range(run_amount):
        # MANUAL LOGGING
        run = wandb.init(project=wb_run, reinit=True, group=group)

        run.config.n_clusters = n_clusters
        run.config.epochs = epochs
        run.config.is_synth = is_synth
        run.config.test_size = test_size

        # LOAD DATA
        if data == "UNOS":
            dm = UNOSDataModule(
                data_dir, batch_size=256, is_synth=is_synth, test_size=test_size
            )
        if data == "U2U":
            dm = UNOS2UKRegDataModule(
                data_dir,
                batch_size=256,
                is_synth=is_synth,
                control=control,
                test_size=test_size,
            )
        else:
            dm = UKRegDataModule(
                data_dir, batch_size=256, is_synth=is_synth, test_size=test_size
            )
        dm.prepare_data()
        dm.setup(stage="fit")

        cm_kwargs = {
            "k": n_clusters,
            "x_col": dm.x_cols,
            "y_col": "Y",
            "H": {
                "RFR": (RandomForestRegressor, {}),
                "SVR": (svm.SVR, {}),
                "MLPR": (
                    MLPRegressor,
                    {"hidden_layer_sizes": (30, 100, 100, 30), "max_iter": epochs},
                ),
            },
        }

        # CONSTRUCT MODEL
        cm = ConfidentMatch(data=dm._train_processed, o_col=dm.o_cols, **cm_kwargs)

        # TRAIN NETWORK
        cm._get_partitions()
        cm._train()

        # TEST NETWORK
        dm.setup(stage="test")
        X, O, Y, _ = dm.test_dataloader().dataset.tensors
        X = torch.cat((X, O), dim=1).numpy()
        Y = Y.numpy()

        Y_ = np.array([cm.estimate(x) for x in X])

        rmse = np.sqrt(mean_squared_error(Y_, Y))

        # scale
        Y_ = Y_ * dm.std + dm.mean
        Y = Y * dm.std + dm.mean

        diff = np.abs(Y_ - Y).mean()

        run.log({"test - RMSE": rmse, "test - mean diff in days": diff})

        # SAVING
        cm.save(location=run.dir)

    wandb.finish()


if __name__ == "__main__":
    train()
