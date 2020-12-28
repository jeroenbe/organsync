import click, wandb, torch, math

import numpy as np

from torch import nn, optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import MeanSquaredError

from src.data.data_module import UNOSDataModule, UKRegDataModule, UNOS2UKRegDataModule



class OrganSync_Network(pl.LightningModule):
    # INFO: In OrganSync, this network provides
    #   the non-linear representation u. In the
    #   representation space, U, we search for 
    #   synthetic controls used for ITE predictions
    #   and survival analysis.

    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            lr, gamma, weight_decay,
            num_hidden_layers: int=1,
            activation_type='relu',
            dropout_prob: float=.0):

        super().__init__()

        self.lr = lr
        self.gamma = gamma
        self.weight_decay = weight_decay

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        activation_functions = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU
        }
        activation = activation_functions[activation_type]

        hidden_layers = np.array([(
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            activation(),
            nn.Dropout(p=dropout_prob)) for _ in range(num_hidden_layers)]).flatten()

        self.representation = nn.Sequential(                # This predicts u
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            *hidden_layers,
            nn.Linear(self.hidden_dim, self.output_dim),
            activation()
        )

        self.beta = nn.Linear(self.output_dim, 1)           # This predicts Y

        self.loss = MeanSquaredError()
        self.save_hyperparameters()


    # ~~~~~~~~~~~~~
    # TORCH METHODS

    def forward(self, x):
        x = self.representation(x)
        x = self.beta(x)

        return x



    # ~~~~~~~~~~~
    # LIT METHODS

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimiser, self.gamma)

        return [optimiser], [scheduler]
    
    
    def shared_step(self, batch):
        x, o, y, _ = batch

        u = torch.cat((x,o), dim=1)
        y_ = self.forward(u)

        loss = self.loss(y_, y)

        return loss
    
    def training_step(self, batch, ix):
        loss = self.shared_step(batch)

        self.log('train_loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, ix):
        loss = self.shared_step(batch)

        self.log('val_loss', loss, on_epoch=True)

        return loss

    def test_step(self, batch, ix):
        x, o, y, _ = batch
        y = y.cpu()
        
        # PREDICT
        u = torch.cat((x, o), dim=1)
        y_ = self.forward(u).cpu()

        # SCALE
        mean, std = self.trainer.datamodule.mean, self.trainer.datamodule.std
        y = y * std + mean
        y_ = y_ * std + mean

        loss = torch.abs(y - y_)

        self.log('test_loss - mean difference in days', loss, on_epoch=True)

        return loss


@click.command()
@click.option('--lr', type=float, default=.005)
@click.option('--gamma', type=float, default=.9)
@click.option('--weight_decay', type=float, default=1e-3)
@click.option('--epochs', type=int, default=30)
@click.option('--wb_run', type=str, default='organsync-net')
@click.option('--batch_size', type=int, default=128)
@click.option('--group', type=str, default=None)
@click.option('--data', type=str, default='UNOS')
@click.option('--data_dir', type=str, default='./data/processed')
@click.option('--num_hidden_layers', type=int, default=1)
@click.option('--output_dim', type=int, default=8)
@click.option('--hidden_dim', type=int, default=16)
@click.option('--activation_type', type=str, default='relu') # 'relu', 'leaky_relu'
@click.option('--dropout_prob', type=float, default=.0)
def train(
        lr,
        gamma,
        weight_decay,
        epochs,
        wb_run,
        batch_size,
        group,
        data,
        data_dir,
        num_hidden_layers,
        output_dim,
        hidden_dim,
        activation_type,
        dropout_prob):

    # LOAD DATA
    if data == 'UNOS':
        dm = UNOSDataModule(data_dir, batch_size=batch_size)
    if data == 'U2U':
        dm = UNOS2UKRegDataModule(data_dir, batch_size=batch_size)
    else:
        dm = UKRegDataModule(data_dir, batch_size=batch_size)
    #dm.setup(stage='fit')

    # CONSTRUCT MODEL
    input_dim = dm.size(1)
    model = OrganSync_Network(
        input_dim=input_dim, 
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        output_dim=output_dim, 
        lr=lr, gamma=gamma, weight_decay=weight_decay,
        activation_type=activation_type,
        dropout_prob=dropout_prob).double()

    # SETUP LOGGING CALLBACKS
    wb_logger = WandbLogger(project=wb_run, log_model=True, group=group)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', filename='organsync_net.ckpt', dirpath=wb_logger.experiment.dir)

    # SETUP GPU
    gpus = 1 if torch.cuda.is_available() else 0

    # TRAIN NETWORK
    trainer = Trainer(logger=wb_logger, callbacks=[checkpoint_callback], max_epochs=epochs, gpus=gpus)
    trainer.fit(model, datamodule=dm)

    # TEST NETWORK
    trainer.test(datamodule=dm)
    
    wandb.finish()

if __name__ == "__main__":
    train()
