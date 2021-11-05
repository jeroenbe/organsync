from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from organsync.data.data_module import OrganDataModule


@dataclass
class Organ:
    id: int
    covariates: np.array

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Organ):
            return NotImplemented
        return self.id == other.id

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Organ):
            return NotImplemented
        return self.id > other.id


@dataclass
class Patient:
    id: int
    covariates: np.array

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Patient):
            return NotImplemented
        return self.id == other.id

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Patient):
            return NotImplemented
        return self.id > other.id


class Policy(ABC):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information
        #   of the transplant system
    ) -> None:

        self.name = name
        self.waitlist = initial_waitlist
        self.dm = dm
        self.test = dm._test_processed  # always perform on test set

    @abstractclassmethod
    def get_xs(self, organs: np.ndarray) -> np.ndarray:
        # Given the internal state of the transplant system
        # waitlist, and a new organ, a patient is suggested.
        # For each patient the ID is used/returned; the policy may
        # use dm for full covariates. When the patient is presented
        # they should be removed from the waitlist.
        #
        # params -
        # :organ: int - organ ID, for reference to dm.ID (note,
        #   dm.ID covers patient-organ pair)
        ...

    @abstractclassmethod
    def add_x(self, x: np.ndarray) -> None:
        # Whenever a patient enters the transplant system
        # add_x is called. It is the policies task to maintain
        # system state.
        #
        # params -
        # :x: int - patient ID, for reference to dm.ID (note,
        #   dm.ID covers patient-organ pair)
        ...

    @abstractclassmethod
    def remove_x(self, x: np.ndarray) -> None:
        # Removes x from the waitlist; happens when they
        # died prematurely. It is the Sim's responsibility
        # to define when a patients dies. As long as the patient
        # remains on the waitlist, they are assumed to be alive.
        #
        # params -
        # :x: int - patient ID, for reference to dm.ID (note,
        #   dm.ID covers patient-organ pair)
        ...

    def waitlist_ids(self) -> np.ndarray:
        return np.array([p.id for p in self.waitlist])
