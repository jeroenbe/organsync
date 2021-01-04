import click, wandb, torch, math

import numpy as np
import cvxpy as cp

import joblib
from joblib import Parallel, delayed

from sklearn.cluster import KMeans

from torch import nn, optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import MeanSquaredError

from src.data.data_module import UNOSDataModule, UKRegDataModule, UNOS2UKRegDataModule
from src.models.utils import GradientReversal

class OrganITE_Network(pl.LightningModule):

    # TODO x: learn cluster in self.setup(stage)
    # TODO x: on_load_checkpoint(self, checkpoint) & on_save_checkpoint(self, checkpoint)
    # TODO: perhaps write small API to efficiently infer
    #   from self.cluster
    # TODO: train this net + copy and train ConfidentMatch
    # TODO: train simple multitask on cluster

    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            lr, gamma, lambd, kappa, weight_decay, n_clusters,
            num_hidden_layers: int=1,
            activation_type='relu',
            dropout_prob: float=.0):

        super().__init__()

        self.lr = lr
        self.gamma = gamma
        self.lambd = lambd
        self.kappa = kappa
        self.weight_decay = weight_decay

        self.n_clusters = n_clusters

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim # phi


        activation_functions = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU
        }
        activation = activation_functions[activation_type]

        hidden_layers = np.array([(
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            activation(),
            nn.Dropout(p=dropout_prob)) for _ in range(num_hidden_layers)]).flatten()

        
        # NETWORK
        self.representation = nn.Sequential(                # This predicts u
            nn.Linear(self.input_dim, self.hidden_dim),
            activation(),
            *hidden_layers,
            nn.Linear(self.hidden_dim, self.output_dim),
            activation()
        )

        self.propensity = nn.Sequential( # p(c(O) | Phi)
            GradientReversal(self.lambd),
            nn.Linear(output_dim, output_dim),
            activation(),
            nn.Linear(output_dim, self.n_clusters),
            nn.Sigmoid()
        )

        self.output = nn.Sequential( # Y | Phi
            nn.Linear(output_dim, output_dim),
            activation(),
            nn.Linear(output_dim, 1)
        )

        self.loss_mse = MeanSquaredError()
        self.loss_cel = nn.CrossEntropyLoss()

        self.save_hyperparameters()


    # ~~~~~~~~~~~~~
    # TORCH METHODS

    def forward(self, x):
        phi = self.representation(x)
        p = self.propensity(phi)
        y = self.output(phi)

        return p, y



    # ~~~~~~~~~~~
    # LIT METHODS

    def on_fit_start(self, stage=None):
        self.cluster = KMeans(n_clusters=self.n_clusters)
        _, O, _, _ = self.trainer.datamodule.train_dataloader().dataset.dataset.tensors

        self.cluster.fit(O)
        
        self.log('cluster sizes', self.cluster.labels_)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['cluster'] = self.cluster
    
    def on_load_checkpoint(self, checkpoint):
        self.cluster = checkpoint['cluster']

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimiser, self.gamma)

        return [optimiser], [scheduler]
    
    # INFERENCE

    def shared_step(self, batch):
        x, o, y, _ = batch

        u = torch.cat((x,o), dim=1)
        p, y_ = self.forward(u)

        c = self.cluster.predict(o.cpu())
        c = torch.Tensor(c).to(self.device).long()

        mse = self.loss_mse(y_, y)
        prop = self.loss_cel(p, c)

        return mse, prop # scale prop
    
    def training_step(self, batch, ix):
        mse, prop = self.shared_step(batch)
        loss = mse + self.kappa * prop

        self.log('train_loss - MSE', mse, on_epoch=True)
        self.log('train_loss - Prop.', prop, on_epoch=True)
        self.log('train_loss - total', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, ix):
        mse, prop = self.shared_step(batch)
        loss = mse + self.kappa * prop

        self.log('val_loss - MSE', mse, on_epoch=True)
        self.log('val_loss - Prop.', prop, on_epoch=True)
        self.log('val_loss - total', loss, on_epoch=True)

        return loss

    def test_step(self, batch, ix):
        x, o, y, _ = batch
        
        _, y_ = self.forward(torch.cat((x, o), dim=1))

        rmse = torch.sqrt(self.loss_mse(y_, y))

        # SCALE
        mean, std = self.trainer.datamodule.mean, self.trainer.datamodule.std
        y = y * std + mean
        y_ = y_ * std + mean

        loss = torch.abs(y - y_)
        self.log('test_loss - mean difference in days', loss, on_epoch=True)
        self.log('test_loss - RMSE', rmse, on_epoch=True)

        return loss



@click.command()
@click.option('--lr', type=float, default=.005)
@click.option('--gamma', type=float, default=.9)
@click.option('--lambd', type=float, default=.15)
@click.option('--kappa', type=float, default=.15)
@click.option('--n_clusters', type=int, default=15)
@click.option('--weight_decay', type=float, default=1e-3)
@click.option('--epochs', type=int, default=30)
@click.option('--wb_run', type=str, default='organsync-organite-net')
@click.option('--run_amount', type=int, default=1)
@click.option('--batch_size', type=int, default=128)
@click.option('--group', type=str, default=None)
@click.option('--data', type=str, default='UNOS')
@click.option('--data_dir', type=str, default='./data/processed')
@click.option('--num_hidden_layers', type=int, default=1)
@click.option('--output_dim', type=int, default=8)
@click.option('--hidden_dim', type=int, default=16)
@click.option('--activation_type', type=str, default='relu') # 'relu', 'leaky_relu'
@click.option('--dropout_prob', type=float, default=.0)
@click.option('--control', type=click.BOOL, default=False)
@click.option('--is_synth', type=click.BOOL, default=False)
@click.option('--test_size', type=float, default=.05)
def train(
        lr,
        gamma,
        lambd,
        kappa,
        n_clusters,
        weight_decay,
        epochs,
        wb_run,
        run_amount,
        batch_size,
        group,
        data,
        data_dir,
        num_hidden_layers,
        output_dim,
        hidden_dim,
        activation_type,
        dropout_prob,
        control,
        is_synth,
        test_size):

    for _ in range(run_amount):
        # LOAD DATA
        if data == 'UNOS':
            dm = UNOSDataModule(data_dir, batch_size=batch_size, is_synth=is_synth, test_size=test_size)
        if data == 'U2U':
            dm = UNOS2UKRegDataModule(data_dir, batch_size=batch_size, is_synth=is_synth, control=control, test_size=test_size)
        else:
            dm = UKRegDataModule(data_dir, batch_size=batch_size, is_synth=is_synth, test_size=test_size)
        #dm.setup(stage='fit')

        # CONSTRUCT MODEL
        input_dim = dm.size(1)
        model = OrganITE_Network(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            output_dim=output_dim, 
            lr=lr, gamma=gamma, lambd=lambd, kappa=kappa, weight_decay=weight_decay,
            n_clusters=n_clusters,
            activation_type=activation_type,
            dropout_prob=dropout_prob).double()

        # SETUP LOGGING CALLBACKS
        wb_logger = WandbLogger(project=wb_run, log_model=True, group=group)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss - MSE', filename='organite_net.ckpt', dirpath=wb_logger.experiment.dir)

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