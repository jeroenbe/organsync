import multiprocessing
from typing import Any

import cvxpy as cp
import numpy as np
import pytorch_lightning as pl
import torch
from joblib import Parallel, delayed
from torch import nn, optim
from torchmetrics import MeanSquaredError

THREADS = multiprocessing.cpu_count()
DISPATCHER = Parallel(n_jobs=THREADS)


# CONVEX OPT
def convex_opt(
    u: np.ndarray, U: np.ndarray, lambd: float, Y: torch.Tensor, solver: Any = cp.SCS
) -> tuple:
    a = cp.Variable(U.shape[0])

    cost = cp.sum_squares(a @ U - u) + lambd * cp.norm1(a)
    objective = cp.Minimize(cost)
    constraints = [0 <= a, a <= 1, cp.sum(a) == 1]

    prob = cp.Problem(objective, constraints)

    prob.solve(warm_start=True, solver=solver)

    return a.value, a.value @ U, (a.value @ Y.numpy()).squeeze()


class OrganSync_Network(pl.LightningModule):
    # INFO: In OrganSync, this network provides
    #   the non-linear representation u. In the
    #   representation space, U, we search for
    #   synthetic controls used for ITE predictions
    #   and survival analysis. For this, refer
    #   to synthetic_control(x, o, lambd)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lr: float,
        gamma: float,
        lambd: float,
        weight_decay: float,
        num_hidden_layers: int = 1,
        activation_type: str = "relu",
        dropout_prob: float = 0.0,
        control_size: int = 1500,
    ) -> None:

        super().__init__()

        self.lr = lr
        self.gamma = gamma
        self.lambd = lambd
        self.weight_decay = weight_decay

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.control_size = control_size

        activation_functions = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}
        activation = activation_functions[activation_type]

        hidden_layers = np.array(
            [
                (
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    activation(),
                    nn.Dropout(p=dropout_prob),
                )
                for _ in range(num_hidden_layers)
            ]
        ).flatten()

        self.representation = nn.Sequential(  # This predicts u
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            *hidden_layers,
            nn.Linear(self.hidden_dim, self.output_dim),
            activation(),
        ).double()

        self.beta = nn.Linear(self.output_dim, 1).double()  # This predicts Y

        self.loss = MeanSquaredError()
        self.save_hyperparameters()

    # ~~~~~~~~~~~~~
    # TORCH METHODS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.representation(x)
        x = self.beta(x)

        return x

    # ~~~~~~~~~~~
    # LIT METHODS

    def on_fit_start(self, stage: Any = None) -> None:
        self.rescale_mean, self.rescale_std = (
            self.trainer.datamodule.mean,
            self.trainer.datamodule.std,
        )
        indices = torch.randint(
            0, len(self.trainer.datamodule._train_processed), (self.control_size,)
        )
        X, O, Y, _ = self.trainer.datamodule.train_dataloader().dataset.dataset[indices]
        catted = torch.cat((X, O), dim=1).double()
        if torch.cuda.is_available():
            catted = catted.cuda()

        self.synth_control_data = (catted, Y, indices)

    def configure_optimizers(self) -> tuple:
        optimiser = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimiser, self.gamma)

        return [optimiser], [scheduler]

    def shared_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, o, y, _ = batch

        u = torch.cat((x, o), dim=1)
        y_ = self.forward(u)

        loss = self.loss(y_, y)

        return loss

    def training_step(self, batch: torch.Tensor, ix: int) -> torch.Tensor:
        loss = self.shared_step(batch)

        self.log("train_loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple, ix: int) -> torch.Tensor:
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch: tuple, ix: int) -> torch.Tensor:
        x, o, y, _ = batch
        y = y

        # PREDICT
        u = torch.cat((x, o), dim=1)
        y_ = self.forward(u)

        # SYNTHETIC PREDICTION
        synth_result = self.synthetic_control(x, o)
        synth_y_scaled = (
            torch.Tensor(synth_result[2].astype("float64")).to(self.device).view(-1, 1)
        )
        synth_y = (
            torch.Tensor(synth_result[3].astype("float64")).to(self.device).view(-1, 1)
        )

        rmse = torch.sqrt(self.loss(y_, y))
        synth_rmse = torch.sqrt(self.loss(synth_y, y))

        # SCALE
        mean, std = self.rescale_mean, self.rescale_std
        y = y * std + mean
        y_ = y_ * std + mean

        loss = torch.abs(y - y_)
        synth_loss = torch.abs(y - synth_y_scaled)

        self.log("test_loss (reg.) - mean difference in days", loss, on_epoch=True)
        self.log(
            "test_loss (synth) - mean difference in days", synth_loss, on_epoch=True
        )

        self.log("test_loss (reg.) - RMSE", rmse, on_epoch=True)
        self.log("test_loss (synth) - RMSE", synth_rmse, on_epoch=True)

        return loss, synth_loss, rmse, synth_rmse

    def synthetic_control(
        self, x: torch.Tensor, o: torch.Tensor, solver: Any = cp.SCS
    ) -> tuple:  # returns a, u_ and y_
        # BUILD U FROM TRAINING
        synth_catted, Y, indices = self.synth_control_data
        U = self.representation(synth_catted).detach().cpu().numpy()

        # BUILD u FROM TEST
        new_pairs = torch.cat((x, o), dim=1).double()
        u = self.representation(new_pairs).detach().cpu().numpy()

        result = np.array(
            DISPATCHER(
                delayed(convex_opt)(chunk, U, lambd=self.lambd, Y=Y, solver=solver)
                for chunk in u
            )
        )

        # INFER
        a_s = result[:, 0]
        u_s = result[:, 1]
        synth_y = result[:, 2]

        mean, std = self.rescale_mean, self.rescale_std
        synth_y_scaled = synth_y * std + mean

        return a_s, u_s, synth_y_scaled, synth_y, indices
