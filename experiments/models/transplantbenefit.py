import click
import numpy as np
import wandb
from sklearn.metrics import mean_squared_error

# OWN MODULES
from experiments.data.data_module import (
    UKRegDataModule,
    UNOS2UKRegDataModule,
    UNOSDataModule,
)
from organsync.models.transplantbenefit import UKELDModel


@click.command()
@click.option("--wb_run", type=str, default="organsync-tb-u2u")
@click.option("--run_amount", type=int, default=1)
@click.option("--group", type=str, default=None)
@click.option("--data", type=str, default="U2U")
@click.option("--data_dir", type=str, default="./data/processed_UNOS2UKReg_no_split")
@click.option("--is_synth", type=click.BOOL, default=False)
@click.option("--test_size", type=float, default=0.05)
@click.option("--penalizer", type=float, default=0.1)
@click.option("--control", type=bool, default=False)
def train(
    wb_run: str,
    run_amount: int,
    group: str,
    data: str,
    data_dir: str,
    is_synth: bool,
    test_size: float,
    penalizer: float,
    control: bool,
) -> None:

    for _ in range(run_amount):
        # MANUAL LOGGING
        run = wandb.init(project=wb_run, reinit=True, group=group)

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
                test_size=test_size,
                control=control,
            )
        else:
            dm = UKRegDataModule(
                data_dir,
                batch_size=256,
                is_synth=is_synth,
                test_size=test_size,
                control=control,
            )
        dm.prepare_data()
        dm.setup(stage="fit")

        # CONSTRUCT MODEL
        DATA = dm._train_processed
        DATA.CENS = np.abs(DATA.CENS - 1)

        cols = np.union1d(dm.x_cols, dm.o_cols)
        if control:
            cols = dm.x_cols
        cols = cols[cols != "CENS"]
        ukeld = UKELDModel(
            data=DATA,
            cols=cols,
            censor_col="CENS",
            duration_col="Y",
            penalizer=penalizer,
        )

        # TRAIN MODEL
        ukeld.fit()

        # TEST NETWORK
        dm.setup(stage="test")
        DATA_t = dm._test_processed
        X = DATA_t[cols].to_numpy()
        Y = DATA_t.Y.to_numpy()

        Y_ = ukeld.estimate(X)
        Y_ = Y_.to_numpy().reshape(-1, 1)

        rmse = np.sqrt(mean_squared_error(Y_, Y))

        # scale
        Y_ = Y_ * dm.std + dm.mean
        Y = Y * dm.std + dm.mean

        diff = np.abs(Y_ - Y).mean()

        run.log({"test - RMSE": rmse, "test - mean diff in days": diff})

        # SAVING
        ukeld.save_cph(location=run.dir, name=f"{data}_cph")

    wandb.finish()


if __name__ == "__main__":
    train()
