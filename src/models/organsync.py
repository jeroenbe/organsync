import click, wandb, torch, math

import numpy as np
import cvxpy as cp

import joblib
from joblib import Parallel, delayed

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
    #   and survival analysis. For this, refer 
    #   to synthetic_control(x, o, lambd)

    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            lr, gamma, lambd, weight_decay,
            num_hidden_layers: int=1,
            activation_type='relu',
            dropout_prob: float=.0):

        super().__init__()

        self.lr = lr
        self.gamma = gamma
        self.lambd = lambd
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
        y = y

        # PREDICT
        u = torch.cat((x, o), dim=1)
        y_ = self.forward(u)

        # SYNTHETIC PREDICTION
        synth_result = self.synthetic_control(x, o)
        synth_y_scaled = torch.Tensor(synth_result[2].astype('float64')).to(self.device).view(-1, 1)
        synth_y = torch.Tensor(synth_result[3].astype('float64')).to(self.device).view(-1, 1)


        rmse = torch.sqrt(self.loss(y_, y))
        synth_rmse = torch.sqrt(self.loss(synth_y, y))

        # SCALE
        mean, std = self.trainer.datamodule.mean, self.trainer.datamodule.std
        y = y * std + mean
        y_ = y_ * std + mean

        loss = torch.abs(y - y_)
        synth_loss = torch.abs(y - synth_y_scaled)

        self.log('test_loss (reg.) - mean difference in days', loss, on_epoch=True)
        self.log('test_loss (synth) - mean difference in days', synth_loss, on_epoch=True)

        self.log('test_loss (reg.) - RMSE', rmse, on_epoch=True)
        self.log('test_loss (synth) - RMSE', synth_rmse, on_epoch=True)

        return loss, synth_loss, rmse, synth_rmse

    def synthetic_control(self, x, o): # returns a, u_ and y_
        # BUILD U FROM TRAINING
        X, O, Y, _ = self.trainer.datamodule.train_dataloader().dataset.dataset[:1000]
        catted = torch.cat((X, O), dim=1).double()
        if torch.cuda.is_available():
            catted = catted.cuda()
        
        U = self.representation(catted).detach().cpu().numpy()

        # BUILD u FROM TEST
        new_pairs = torch.cat((x, o), dim=1).double()
        u = self.representation(new_pairs).detach().cpu().numpy()
        
        # CONVEX OPT
        def convex_opt(u):
            a = cp.Variable(U.shape[0])

            objective = cp.Minimize(cp.sum_squares(a@U - u) + self.lambd * cp.norm1(a))
            constraints = [0 <= a, a <= 1, cp.sum(a) == 1]
            prob = cp.Problem(objective, constraints)

            prob.solve(warm_start=True, solver=cp.SCS)
            
            return a.value, a.value @ U, (a.value @ Y.numpy()).item()

        #result = Parallel(n_jobs=int(joblib.cpu_count()/2))(delayed(convex_opt)(u_) for u_ in u)
        result = np.array([convex_opt(u_) for u_ in u])
        result = np.array(result, dtype=object)
        
        # INFER
        a_s = result[:,0]
        u_s = result[:,1]
        synth_y = result[:,2]

        mean, std = self.trainer.datamodule.mean, self.trainer.datamodule.std
        synth_y_scaled = synth_y * std + mean

        return a_s, u_s, synth_y_scaled, synth_y


@click.command()
@click.option('--lr', type=float, default=.005)
@click.option('--gamma', type=float, default=.9)
@click.option('--lambd', type=float, default=.5)
@click.option('--weight_decay', type=float, default=1e-3)
@click.option('--epochs', type=int, default=30)
@click.option('--wb_run', type=str, default='organsync-net')
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
        elif data == 'U2U':
            dm = UNOS2UKRegDataModule(data_dir, batch_size=batch_size, is_synth=is_synth, control=control, test_size=test_size)
        else:
            dm = UKRegDataModule(data_dir, batch_size=batch_size, is_synth=is_synth, test_size=test_size)
        #dm.setup(stage='fit')

        # CONSTRUCT MODEL
        input_dim = dm.size(1)
        model = OrganSync_Network(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            output_dim=output_dim, 
            lr=lr, gamma=gamma, lambd=lambd, weight_decay=weight_decay,
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

        wandb.run.join()
    
    wandb.finish()

if __name__ == "__main__":
    train()
