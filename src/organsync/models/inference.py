from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from organsync.data.data_module import OrganDataModule
from organsync.models.confidentmatch import ConfidentMatch
from organsync.models.organite_network import OrganITE_Network, OrganITE_Network_VAE
from organsync.models.organsync_network import OrganSync_Network
from organsync.models.transplantbenefit import TBS


class Inference(ABC):
    def __init__(self, model: Any, mean: float, std: float) -> None:
        assert isinstance(mean, (float, int)), "mean must be float or int"
        assert isinstance(std, (float, int)), "std must be float or int"
        self.model = model
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.infer(x, *args, **kwargs)

    @abstractmethod
    def infer(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        ...


class Inference_OrganSync(Inference):
    def __init__(self, model: OrganSync_Network, mean: float, std: float) -> None:
        assert isinstance(model, OrganSync_Network), "Model must be OrganSync_Network"
        super().__init__(model, mean, std)

    def infer(self, x: torch.Tensor, o: Optional[torch.Tensor] = None, SC: bool = False) -> Any:  # type: ignore
        with torch.no_grad():
            x = torch.Tensor(x).double()

            if o is not None:
                o = torch.Tensor(o).double()

            if SC and o is not None:
                a, _, y, _, ixs = self.model.synthetic_control(x, o)
            else:
                a, ixs = None, None
                if o is not None:
                    x = torch.cat((x, o), dim=1)
                y = self.model(x)
                y = y * self.std + self.mean
            return y, a, ixs


class Inference_OrganITE(Inference):
    def __init__(self, model: OrganITE_Network, mean: float, std: float) -> None:
        assert isinstance(model, OrganITE_Network), "Model must be OrganITE_Network"
        super().__init__(model, mean, std)

    def infer(self, x: torch.Tensor, o: Optional[torch.Tensor] = None, replace_organ: int = -1) -> Any:  # type: ignore
        # NOTE: replace_organ should fit what has been defined
        #   at training time
        with torch.no_grad():
            if o is None:
                o = np.full((len(x), len(self.model.o_cols)), replace_organ)
            x = torch.Tensor(x).double()
            o = torch.Tensor(o).double()
            catted = torch.cat((x, o), dim=1)

            _, y = self.model(catted)
            y = y * self.std + self.mean

            return y


class Inference_OrganITE_VAE(Inference):
    def __init__(self, model: OrganITE_Network_VAE, mean: float, std: float) -> None:
        assert isinstance(
            model, OrganITE_Network_VAE
        ), "Model must be OrganITE_Network_VAE"
        super().__init__(model, mean, std)

    def infer(self, x: torch.Tensor) -> Any:  # type: ignore
        with torch.no_grad():
            x = torch.Tensor(x).double()
            prob = self.model.p(x)
            return prob


class Inference_ConfidentMatch(Inference):
    def __init__(self, model: ConfidentMatch, mean: float, std: float) -> None:
        assert isinstance(model, ConfidentMatch), "Model must be ConfidentMatch"
        super().__init__(model, mean, std)

    def infer(self, x: torch.Tensor, o: torch.Tensor) -> Any:  # type: ignore
        # NOTE: ConfidentMatch is not an ITE model, so only estimates
        #   using X AND O together.
        X = np.append(x, o, axis=1)
        return self.model.estimate(X)


class Inference_TBS(Inference):
    def __init__(
        self, model: TBS, mean: float, std: float, dm: OrganDataModule
    ) -> None:
        assert isinstance(model, TBS), "Model must be TBS"
        super().__init__(model, mean, std)

        self.dm = dm

    def infer(self, x: pd.DataFrame) -> Any:
        x = self._transform_data(x)
        res = self.model.predict(data=x)
        return res.score.values, res.m1.values, res.m2.values  # tbs, m1, m2

    def _transform_data(self, x: pd.DataFrame) -> pd.DataFrame:
        x_true = x.copy(deep=True)

        present = np.where(np.in1d(self.dm.real_cols, x_true.columns.values))[0]
        cols = self.dm.real_cols[present]

        x_true.loc[:, cols] = (
            self.dm.scaler.mean_[present]
            + x_true.loc[:, cols] * self.dm.scaler.scale_[present]
        )

        return x_true
