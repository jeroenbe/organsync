from dataclasses import dataclass
from typing import Any

import numpy as np


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
