from typing import Any, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.cluster import KMeans
from torch import nn, optim
from torchmetrics import MeanSquaredError


class MultiTask_Network(pl.LightningModule):
    # TODO: train this net + copy and train ConfidentMatch
    # TODO: train simple multitask on cluster

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lr: float,
        gamma: float,
        weight_decay: float,
        n_clusters: int,
        num_hidden_layers: int = 1,
        activation_type: str = "relu",
        dropout_prob: float = 0.0,
    ) -> None:

        super().__init__()

        self.lr = lr
        self.gamma = gamma
        self.weight_decay = weight_decay

        self.n_clusters = n_clusters

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # phi

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

        # NETWORK
        self.representation = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            activation(),
            *hidden_layers,
            nn.Linear(self.hidden_dim, self.output_dim),
            activation(),
        )

        self.output = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            activation(),
            nn.Linear(output_dim, self.n_clusters),
        )

        self.loss_mse = MeanSquaredError()

        self.save_hyperparameters()

    # ~~~~~~~~~~~~~
    # TORCH METHODS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.representation(x)
        y = self.output(phi)

        return y

    # ~~~~~~~~~~~
    # LIT METHODS

    def on_fit_start(self, stage: Any = None) -> None:
        self.cluster = KMeans(n_clusters=self.n_clusters)
        _, O, _, _ = self.trainer.datamodule.train_dataloader().dataset.dataset.tensors

        self.cluster.fit(O)

        self.log("cluster sizes", self.cluster.labels_)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["cluster"] = self.cluster

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.cluster = checkpoint["cluster"]

    def configure_optimizers(self) -> tuple:
        optimiser = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimiser, self.gamma)

        return [optimiser], [scheduler]

    # ~~~~~~~~~~~
    # INFERENCE

    def shared_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, o, y, _ = batch  # y (batch_size,)

        u = torch.cat((x, o), dim=1)
        y_ = self.forward(u)  # (batch_size, n_clusters)

        c = self.cluster.predict(o.cpu())
        c = torch.Tensor(c).to(self.device).view(-1)
        mask = torch.zeros((len(y), self.n_clusters))

        for i, c_ in enumerate(c.long()):
            mask[i, c_] = 1

        new_y = y_.clone().detach()  # loss = 0 on non-observed tasks
        new_y[mask.bool()] = y.view(-1)

        mse = self.loss_mse(y_, new_y)

        return mse

    def training_step(self, batch: torch.Tensor, ix: int) -> torch.Tensor:
        loss = self.shared_step(batch)

        self.log("train_loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch: torch.Tensor, ix: int) -> torch.Tensor:
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_epoch=True)

        return loss

    def test_step(
        self, batch: torch.Tensor, ix: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.log("test_loss - mean difference in days", loss, on_epoch=True)
        self.log("test_loss - RMSE", rmse, on_epoch=True)

        return loss, rmse
