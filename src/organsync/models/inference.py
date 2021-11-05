from abc import ABC, abstractclassmethod
from typing import Any, Optional, Tuple

import numpy as np
import torch

from organsync.models.confidentmatch import ConfidentMatch
from organsync.models.organite_network import OrganITE_Network, OrganITE_Network_VAE
from organsync.models.organsync_network import OrganSync_Network
from organsync.models.transplantbenefit import UKELDModel


class Inference(ABC):
    def __init__(self, model: Any, mean: float, std: float) -> None:
        assert isinstance(mean, (float, int)), "mean must be float or int"
        assert isinstance(std, (float, int)), "std must be float or int"
        self.model = model
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.infer(x, *args, **kwargs)

    @abstractclassmethod
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
                a, _, y, _, ixs = self.model.synthetic_control(x, o, n=1500)
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
                o = np.full(
                    (len(x), len(self.model.trainer.datamodule.o_cols)), replace_organ
                )
            x = torch.Tensor(x).double()
            o = torch.Tensor(o).double()
            catted = torch.cat((x, o), dim=1)

            _, y = self.model(catted)

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


class Inference_TransplantBenefit(Inference):
    def __init__(
        self, model: Tuple[UKELDModel, UKELDModel], mean: float, std: float
    ) -> None:
        assert isinstance(model, tuple), "Expecting two UKELDModel models in input"
        assert isinstance(model[0], UKELDModel), "Expecting first model UKELDModel"
        assert isinstance(
            model[1], UKELDModel
        ), "Expecting second model UKELDModel in input"

        super().__init__(model, mean, std)
        self.model_0 = model[0]
        self.model_1 = model[1]

    def infer(self, x: torch.Tensor, o: torch.Tensor) -> Any:  # type: ignore
        X = np.append(x, o, axis=1)
        X_0 = np.append(x, np.zeros(np.array(o).shape), axis=1)
        Y_1 = self.model_1.estimate(X)
        Y_0 = self.model_0.estimate(X_0)
        return Y_1 - Y_0
