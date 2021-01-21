import numpy as np
import pandas as pd

from abc import ABC
from abc import abstractclassmethod

from dataclasses import dataclass

from src.data.data_module import OrganDataModule

from operator import attrgetter

@dataclass
class Organ:
    id: int
    covariates: np.array

@dataclass
class Patient:
    id: int
    covariates: np.array

    def __eq__(self, other):
        if not isinstance(other, Patient):
            return NotImplemented
        return self.id == other.id
    
    def __gt__(self, other):
        if not isinstance(other, Patient):
            return NotImplemented
        return self.id > other.id


class Policy(ABC):
    def __init__(self, 
            name: str,                      # policy name, reported in wandb
            initial_waitlist: np.array,     # waitlist upon starting the simulation, [int]
            dm: OrganDataModule,            # datamodule containing all information 
                                            #   of the transplant system
        ):
        
        self.name = name
        self.waitlist = initial_waitlist
        self.dm = dm
        self.test = dm._test_processed # always perform on test set
    
    @abstractclassmethod
    def get_xs(self, organ: int) -> int:
        # Given the internal state of the transplant system
        # waitlist, and a new organ, a patient is suggested.
        # For each patient the ID is used/returned; the policy may 
        # use dm for full covariates. When the patient is presented
        # they should be removed from the waitlist.
        #
        # params -
        # :organ: int - organ ID, for reference to dm.ID (note, 
        #   dm.ID covers patient-organ pair)
        pass

    @abstractclassmethod
    def add_x(self, x: int):
        # Whenever a patient enters the transplant system
        # add_x is called. It is the policies task to maintain
        # system state.
        #
        # params - 
        # :x: int - patient ID, for reference to dm.ID (note,
        #   dm.ID covers patient-organ pair)
        pass
    
    @abstractclassmethod
    def remove_x(self, x: int):
        # Removes x from the waitlist; happens when they
        # died prematurely. It is the Sim's responsibility
        # to define when a patients dies. As long as the patient
        # remains on the waitlist, they are assumed to be alive.
        #
        # params - 
        # :x: int - patient ID, for reference to dm.ID (note,
        #   dm.ID covers patient-organ pair)
        pass

    def waitlist_ids(self):
        return np.array([p.id for p in self.waitlist])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# POLICY DEFINITIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# LivSim Policies (MELD and MELD-na)    
class MELD(Policy):
    def __init__(self, name, initial_waitlist, dm):
        super().__init__(name, initial_waitlist, dm)

        self._setup()
    
    def _get_x(self, organ):
        X = max(self.waitlist, key=attrgetter('covariates'))
        self.waitlist = np.delete(self.waitlist, np.where(np.array([p.id for p in self.waitlist]) == X.id)[0].item())

        return X.id
    
    def get_xs(self, organs):
        return np.array([self._get_x(organ) for organ in organs])

    def add_x(self, x):
        MELD_score = self._meld(x)
        
        X = [Patient(id=x[i], covariates=MELD_score[i]) for i in range(len(x))]

        self.waitlist = np.append(self.waitlist, X)
        self.waitlist = np.unique(self.waitlist)        # make sure there no patient is containted twice
    
    def _setup(self):
        MELD_score = self._meld(self.waitlist)

        self.waitlist = np.array([Patient(id=self.waitlist[i], covariates=MELD_score[i]) for i in range(len(self.waitlist))])

    def _meld(self, patients):
        # params - 
        # :patients: array(int) - list of patient IDs

        ps = self.test.loc[self.test.index.isin(patients)].copy()
        ps.loc[:, self.dm.real_cols] = self.dm.scaler.inverse_transform(ps[self.dm.real_cols])

        # DEFINITION OF (standard) MELD: https://en.wikipedia.org/wiki/Model_for_End-Stage_Liver_Disease#Determination
        # FOR MELD-na: MELD+1.59*(135-SODIUM(mmol/l)): https://github.com/kartoun/meld-plus/raw/master/MELD_Plus_Calculator.xlsx
        MELD_score = 3.79 * np.log(ps.SERUM_BILIRUBIN) + 11.2 * np.log(ps.INR) + 9.57 * np.log(ps.SERUM_CREATININE) + 6.43
        return  MELD_score.to_numpy()

    def remove_x(self, x):
        self.waitlist = np.delete(self.waitlist, np.where(np.array([p.id for p in self.waitlist]) == x)[0].item())

class MELD_na(MELD):
    def __init__(self, name, initial_waitlist, dm):
        super().__init__(name, initial_waitlist, dm)
    
    def _meld(self, patients):
        # We can simply inherit from MELD as the only part
        # that changes is they way we compute a MELD score 
        # by adding the last term in MELD_score.

        ps = self.test.loc[self.test.index.isin(patients)].copy()
        ps.loc[:, self.dm.real_cols] = self.dm.scaler.inverse_transform(ps[self.dm.real_cols])

        MELD_score = 3.79 * np.log(ps.SERUM_BILIRUBIN) + 11.2 * np.log(ps.INR) + 9.57 * np.log(ps.SERUM_CREATININE) + 6.43 + 1.59 * (135 - ps.SERUM_SODIUM)
        return MELD_score.to_numpy()

