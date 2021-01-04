import click, wandb, torch

import numpy as np

from sklearn.cluster import KMeans

from torch import nn, optim

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import MeanSquaredError

from src.data.data_module import UNOSDataModule, UKRegDataModule, UNOS2UKRegDataModule

class MultiTask_Network(pl.LightningModule):
    # TODO: train this net + copy and train ConfidentMatch
    # TODO: train simple multitask on cluster

    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            lr, gamma, weight_decay, 
            n_clusters,
            num_hidden_layers: int=1,
            activation_type='relu',
            dropout_prob: float=.0):

        super().__init__()

        self.lr = lr
        self.gamma = gamma
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
        self.representation = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            activation(),
            *hidden_layers,
            nn.Linear(self.hidden_dim, self.output_dim),
            activation()
        )

        self.output = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            activation(),
            nn.Linear(output_dim, self.n_clusters)
        )

        self.loss_mse = MeanSquaredError()

        self.save_hyperparameters()


    # ~~~~~~~~~~~~~
    # TORCH METHODS

    def forward(self, x):
        phi = self.representation(x)
        y = self.output(phi)

        return y



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
    
    
    
    # ~~~~~~~~~~~
    # INFERENCE

    def shared_step(self, batch):
        x, o, y, _ = batch # y (batch_size,)

        u = torch.cat((x,o), dim=1)
        y_ = self.forward(u) # (batch_size, n_clusters)

        c = self.cluster.predict(o.cpu())
        c = torch.Tensor(c).to(self.device).view(-1)
        mask = torch.zeros((len(y), self.n_clusters))
        
        for i, c_ in enumerate(c.long()):
            mask[i, c_] = 1

        new_y = y_.clone().detach() # loss = 0 on non-observed tasks
        new_y[mask.bool()] = y.view(-1)

        mse = self.loss_mse(y_, new_y)

        return mse
    
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
        
        y_ = self.forward(torch.cat((x, o), dim=1))
        
        c = self.cluster.predict(o.cpu())
        c = torch.Tensor(c).to(self.device).view(-1)
        mask = torch.zeros((len(y), self.n_clusters))
        
        for i, c_ in enumerate(c.long()):
            mask[i, c_] = 1

        y_ = y_[mask.bool()].view(-1, 1)

        rmse = torch.sqrt(self.loss_mse(y_, y))

        # SCALE
        mean, std = self.trainer.datamodule.mean, self.trainer.datamodule.std
        y = y * std + mean
        y_ = y_ * std + mean

        loss = torch.abs(y - y_)
        self.log('test_loss - mean difference in days', loss, on_epoch=True)
        self.log('test_loss - RMSE', rmse, on_epoch=True)

        return loss, rmse


@click.command()
@click.option('--lr', type=float, default=.01)
@click.option('--gamma', type=float, default=.9)
@click.option('--n_clusters', type=int, default=15)
@click.option('--weight_decay', type=float, default=.0005)
@click.option('--epochs', type=int, default=50)
@click.option('--wb_run', type=str, default='organsync-organite-net')
@click.option('--run_amount', type=int, default=1)
@click.option('--batch_size', type=int, default=256)
@click.option('--group', type=str, default=None)
@click.option('--data', type=str, default='U2U')
@click.option('--data_dir', type=str, default='./data/processed/U2U_no_split')
@click.option('--num_hidden_layers', type=int, default=2)
@click.option('--output_dim', type=int, default=4)
@click.option('--hidden_dim', type=int, default=60)
@click.option('--activation_type', type=str, default='relu') # 'relu', 'leaky_relu'
@click.option('--dropout_prob', type=float, default=.05)
@click.option('--control', type=click.BOOL, default=False)
@click.option('--is_synth', type=click.BOOL, default=False)
@click.option('--test_size', type=float, default=.05)
def train(
        lr,
        gamma,
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
        model = MultiTask_Network(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            output_dim=output_dim, 
            lr=lr, gamma=gamma, weight_decay=weight_decay,
            n_clusters=n_clusters,
            activation_type=activation_type,
            dropout_prob=dropout_prob).double()

        # SETUP LOGGING CALLBACKS
        wb_logger = WandbLogger(project=wb_run, log_model=True, group=group)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', filename='organite_net.ckpt', dirpath=wb_logger.experiment.dir)

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