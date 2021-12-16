import click
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from experiments.data.data_module import (
    UKRegDataModule,
    UNOS2UKRegDataModule,
    UNOSDataModule,
)
from organsync.models.organite_network import OrganITE_Network, OrganITE_Network_VAE


@click.command()
@click.option("--lr", type=float, default=0.005)
@click.option("--gamma", type=float, default=0.9)
@click.option("--lambd", type=float, default=0.15)
@click.option("--kappa", type=float, default=0.15)
@click.option("--n_clusters", type=int, default=15)
@click.option("--weight_decay", type=float, default=1e-3)
@click.option("--epochs", type=int, default=30)
@click.option("--wb_run", type=str, default="organsync-organite-net")
@click.option("--run_amount", type=int, default=1)
@click.option("--batch_size", type=int, default=128)
@click.option("--group", type=str, default=None)
@click.option("--data", type=str, default="UNOS")
@click.option("--data_dir", type=str, default="./data/processed")
@click.option("--num_hidden_layers", type=int, default=1)
@click.option("--output_dim", type=int, default=8)
@click.option("--hidden_dim", type=int, default=16)
@click.option("--activation_type", type=str, default="relu")  # 'relu', 'leaky_relu'
@click.option("--dropout_prob", type=float, default=0.0)
@click.option("--control", type=click.BOOL, default=False)
@click.option("--is_synth", type=click.BOOL, default=False)
@click.option("--test_size", type=float, default=0.05)
def train(
    lr: float,
    gamma: float,
    lambd: float,
    kappa: float,
    n_clusters: int,
    weight_decay: float,
    epochs: int,
    wb_run: str,
    run_amount: int,
    batch_size: int,
    group: str,
    data: str,
    data_dir: str,
    num_hidden_layers: int,
    output_dim: int,
    hidden_dim: int,
    activation_type: str,
    dropout_prob: float,
    control: bool,
    is_synth: bool,
    test_size: float,
) -> None:

    for _ in range(run_amount):
        # LOAD DATA
        if data == "UNOS":
            dm = UNOSDataModule(
                data_dir, batch_size=batch_size, is_synth=is_synth, test_size=test_size
            )
        elif data == "U2U":
            dm = UNOS2UKRegDataModule(
                data_dir,
                batch_size=batch_size,
                is_synth=is_synth,
                control=control,
                test_size=test_size,
            )
        else:
            dm = UKRegDataModule(
                data_dir, batch_size=batch_size, is_synth=is_synth, test_size=test_size
            )
        # dm.setup(stage='fit')

        # CONSTRUCT MODEL
        input_dim = dm.size(1)
        model = OrganITE_Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            output_dim=output_dim,
            lr=lr,
            gamma=gamma,
            lambd=lambd,
            kappa=kappa,
            weight_decay=weight_decay,
            n_clusters=n_clusters,
            activation_type=activation_type,
            dropout_prob=dropout_prob,
        ).double()

        # SETUP LOGGING CALLBACKS
        wb_logger = WandbLogger(project=wb_run, log_model=True, group=group)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss - MSE",
            filename="organite_net.ckpt",
            dirpath=wb_logger.experiment.dir,
        )

        # SETUP GPU
        gpus = 1 if torch.cuda.is_available() else 0

        # TRAIN NETWORK
        trainer = Trainer(
            logger=wb_logger,
            callbacks=[checkpoint_callback],
            max_epochs=epochs,
            gpus=gpus,
        )
        trainer.fit(model, datamodule=dm)

        # TEST NETWORK
        trainer.test(datamodule=dm)

    wandb.finish()


@click.command()
@click.option("--lr", type=float, default=0.005)
@click.option("--gamma", type=float, default=0.9)
@click.option("--weight_decay", type=float, default=1e-3)
@click.option("--epochs", type=int, default=30)
@click.option("--wb_run", type=str, default="organsync-organite-po-net")
@click.option("--batch_size", type=int, default=128)
@click.option("--data", type=str, default="UNOS")
@click.option("--data_dir", type=str, default="./data/processed")
@click.option("--output_dim", type=int, default=8)
@click.option("--hidden_dim", type=int, default=16)
@click.option("--control", type=click.BOOL, default=False)
@click.option("--is_synth", type=click.BOOL, default=False)
@click.option("--test_size", type=float, default=0.05)
def train_vae(
    lr: float,
    gamma: float,
    weight_decay: float,
    epochs: int,
    wb_run: str,
    batch_size: int,
    data: str,
    data_dir: str,
    output_dim: int,
    hidden_dim: int,
    control: bool,
    is_synth: bool,
    test_size: float,
) -> None:

    # LOAD DATA
    if data == "UNOS":
        dm = UNOSDataModule(
            data_dir, batch_size=batch_size, is_synth=is_synth, test_size=test_size
        )
    elif data == "U2U":
        dm = UNOS2UKRegDataModule(
            data_dir,
            batch_size=batch_size,
            is_synth=is_synth,
            control=control,
            test_size=test_size,
        )
    else:
        dm = UKRegDataModule(
            data_dir, batch_size=batch_size, is_synth=is_synth, test_size=test_size
        )
    dm.prepare_data()
    dm.setup(stage="fit")

    # CONSTRUCT MODEL
    input_dim = len(dm.o_cols)
    model = OrganITE_Network_VAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        lr=lr,
        gamma=gamma,
        weight_decay=weight_decay,
    ).double()

    # SETUP LOGGING CALLBACKS
    wb_logger = WandbLogger(project=wb_run, log_model=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="organite_vae_net",
        dirpath=wb_logger.experiment.dir,
    )

    # SETUP GPU
    gpus = 1 if torch.cuda.is_available() else 0

    # TRAIN NETWORK
    trainer = Trainer(
        logger=wb_logger, callbacks=[checkpoint_callback], max_epochs=epochs, gpus=gpus
    )
    trainer.fit(model, datamodule=dm)

    # TEST NETWORK
    trainer.test(datamodule=dm)

    wandb.finish()


if __name__ == "__main__":
    train()
    # train_vae()
