import numpy as np
import pandas as pd
import cvxpy as cp

import torch

from tqdm import tqdm

from dataclasses import dataclass, field

from abc import ABC, abstractclassmethod

from src.data.data_module import OrganDataModule
from src.models.organsync import OrganSync_Network
from src.models.organite import OrganITE_Network, OrganITE_Network_VAE
from src.models.confidentmatch import ConfidentMatch
from src.models.transplantbenefit import UKELDModel

@dataclass
class Sim_Patient:
    id: int
    time_to_live: int

    def age(self, days):
        self.time_to_live -= days
        return self
    
    def __eq__(self, other):
        if not isinstance(other, Sim_Patient):
            return NotImplemented
        return self.id == other.id
    
    def __gt__(self, other):
        if not isinstance(other, Sim_Patient):
            return NotImplemented
        return self.id > other.id

@dataclass
class Stats:
    deaths: int=0
    patients_seen: int=0
    population_life_years: float=0
    transplant_count: int=0
    first_empty_day: int=-1
    patients_transplanted: dict=field(default_factory=dict) # should be a dict of day: np.array w shape (2, n_patients_on_day)
    organs_transplanted: dict=field(default_factory=dict) 
    patients_died: dict=field(default_factory=dict)         # dict of day: np.array

    def __str__(self):
        return f'Deaths: {self.deaths}\nPopulation life-years: {self.population_life_years}\nTransplant count: {self.transplant_count}\nFirst empty day: {self.first_empty_day}'


class Inference(ABC):
    def __init__(self, model, mean, std):
        self.model = model
        self.mean = mean
        self.std = std
    
    def __call__(self, x, *args, **kwargs):
        return self.infer(x, *args, **kwargs)
    
    @abstractclassmethod
    def infer(self, x, *args, **kwargs):
        pass

class Inference_OrganSync(Inference):
    def __init__(self, model: OrganSync_Network, mean, std):
        super().__init__(model, mean, std)
    
    def infer(self, x, o=None, SC=False):
        with torch.no_grad():
            x = torch.Tensor(x).double()

            if SC and o is not None:
                o = torch.Tensor(o).double()
                a, _, y, _, ixs = self.model.synthetic_control(x, o, n=1500)
            else:
                a, ixs = None, None
                if o is not None:
                    o = torch.Tensor(o).double()
                    x = torch.cat((x, o), dim=1)
                y = self.model(x)
                y = y * self.std + self.mean
            return y, a, ixs

class Inference_OrganITE(Inference):
    def __init__(self, model: OrganITE_Network, mean, std):
        super().__init__(model, mean, std)

    def infer(self, x, o=None, replace_organ=-1):
        # NOTE: replace_organ should fit what has been defined
        #   at training time
        with torch.no_grad():
            if o is None:
                o = np.full((len(x), len(self.model.trainer.datamodule.o_cols)), replace_organ)
            x = torch.Tensor(x).double()
            o = torch.Tensor(o).double()
            catted = torch.cat((x, o), dim=1)

            _, y = self.model(catted)

            return y

class Inference_OrganITE_VAE(Inference):
    def __init__(self, model: OrganITE_Network_VAE, mean, std):
        super().__init__(model, mean, std)
    
    def infer(self, x):
        with torch.no_grad():
            x = torch.Tensor(x).double()
            prob = self.model.p(x)
            return prob

class Inference_ConfidentMatch(Inference):
    def __init__(self, model: ConfidentMatch, mean, std):
        super().__init__(model, mean, std)
    
    def infer(self, x, o):
        # NOTE: ConfidentMatch is not an ITE model, so only estimates
        #   using X AND O together.
        X = np.append(x, o, axis=1)
        return self.model.estimate(X)


class Inference_TransplantBenefit(Inference):
    def __init__(self, model: UKELDModel, mean, std):
        super().__init__(model, mean, std)
        self.model_0 = model[0]
        self.model_1 = model[1]

    def infer(self, x, o):
        X = np.append(x, o, axis=1)
        X_0 = np.append(x, np.zeros(np.array(o).shape), axis=1)
        Y_1 = self.model_1.estimate(X)
        Y_0 = self.model_0.estimate(X_0)
        return Y_1 - Y_0



# SIMULATION OVERVIEW:
#   1. -> setup waitlist of patients
#   2. -> setup available organs (amount
#       will be fraction of patients)
#   3. -> shuffle patients df and organs df
#   4. -> iterate over patients:
#       -> sample patient(s)
#       -> sample organ(s)
#       -> remove dead patients from waitlist (also in policy)
#       -> update statistics


class Sim():
    def __init__(
            self,
            dm: OrganDataModule,         
            initial_waitlist_size: int,
            inference_0: Inference,
            inference_1: Inference,
            organ_deficit: float,
            patient_count: int=1000,
            days: int=365,
        ):

        self.dm = dm
        self.DATA = dm._test_processed.copy(deep=True)                  # These are the 'ground truth' datasets for
        self.patients = self.DATA[dm.x_cols]                            # for patients and organs. From these we'll
        self.organs = self.DATA[dm.o_cols]                              # create a fictive event calender.

        self.organ_deficit = organ_deficit                              # For every 1 patient, there will be 
                                                                        # organ_deficit organs (usually less than 1)
        self.days = days                                                # Amount of days the simulation runs (1 year standard)
        self.patient_count = patient_count                              # Max amount of patients, on average this means 
                                                                        # patient_count/days amount per day.


        self.inference_0 = inference_0                                  # These models will function as ground truth
        self.inference_1 = inference_1                                  # outcomes. Note that LSAM and LivSim use simple
                                                                        # linear models. With an Inference class, this 
                                                                        # can be anything.

        self.waitlist = np.array([])                                    # Get's initialized in self._setup. This allows
        self.initial_waitlist_size = initial_waitlist_size              # multiple runs with one Sim object.

        X_tmp = torch.Tensor(self.DATA[self.dm.x_cols].to_numpy())      # Add time to live (ttl) column
        self.DATA.loc[:,'ttl'] = self.inference_0(X_tmp)[0].numpy()     # to DATA using inference_0

        self.log_df = pd.DataFrame(columns=['day', 'patient_id', 'organ_id'])

        self._setup()

    def _setup(self):
        # RESET WAITLIST TO initial_waitlist_size
        patients_on_waitlist_df = self.DATA.sample(self.initial_waitlist_size)
        waitlist_indxs = patients_on_waitlist_df.index

        self.waitlist = np.array([
            Sim_Patient(
                id=waitlist_indxs[i],
                time_to_live=patients_on_waitlist_df.iloc[i].ttl) 
            for i in range(len(waitlist_indxs))
        ])

        # RESET STATS
        self.stats = Stats()

    def simulate(self, policy, log=False) -> Stats:
        # while not stop_critierum
        #   self.iterate(policy)
        have_empty = False
        for day in tqdm(range(self.days)):
            self.iterate(policy, day, log=log)
            if not have_empty and day > 0 and len(self.stats.patients_transplanted[f'{day-1}']) < len(self.stats.organs_transplanted[f'{day-1}']):
                self.stats.first_empty_day = day-1
                have_empty = True

        return self.stats, self.log_df


    def iterate(self, policy, _day, log=False):
        log_dict = []                                                   # init log_dict -> each day a new log is added

        dead_patients = self._remove_dead_patients(policy)              # remove dead patients from waitlist (also in policy)
        amount_died = len(dead_patients)

        if log:
            log_dict.extend([{
                'patient_id': dead_patient, 
                'organ_id': -1,
                'day': _day} for dead_patient in dead_patients])        # add dead patients with organ None to log_dict
            

        patients = self._sample_patients()                              # sample patient(s)
        organs = self._sample_organs()                                  # sample organ(s)
        policy.add_x(patients)                                          # add patient(s) and organ(s) to 
        transplant_patients = policy.get_xs(organs)                     # the policy's internal waitlist
                                                                        # and assign organs to patients

        if len(organs) > len(transplant_patients):
            organs = organs[:len(transplant_patients)]
        #  CALCULATE TTL FOR transplant_patients BY policy
        organs_cov = self.organs.loc[organs].to_numpy()                 # sample organ covariates for sampled indices
        patients_cov = self.patients.loc[                               # sample patient covariates for transplanted indices
            transplant_patients].to_numpy() 
        
        if log:                                                         # add transplantation to log_dict
            log_dict.extend([{
                'patient_id': transplant_patients[r],
                'organ_id': organs[r],
                'day': _day
            } for r in range(len(transplant_patients))])
        
        catted = np.append(patients_cov, organs_cov, axis=1)            # calculate time to live (ttl) with organ using inference_1
        ttl = np.array([self.inference_1(x)[0] for x in catted])

        self._remove_patients(transplant_patients)                      # remove transplanted patients from waitlist
                                                                        # note that transplant_patients are automatically
                                                                        # removed from the policy's internal waitlist


        # UPDATE STATISTICS
        self.stats.deaths += amount_died
        self.stats.population_life_years += ttl.sum()
        self.stats.transplant_count += len(transplant_patients)

        self.stats.patients_died[f'{_day}'] = dead_patients
        self.stats.patients_transplanted[f'{_day}'] = transplant_patients
        self.stats.organs_transplanted[f'{_day}'] = organs

        if log:                                                         # add log_dict to log_df (later to be returned after simulation ends)
            for log_item in log_dict:
                self.log_df = self.log_df.append(log_item, ignore_index=True)


        self._age_patients(days=30)


    def _remove_patients(self, patients):
        self.waitlist = np.delete(self.waitlist, 
        np.intersect1d(
            np.array([p.id for p in self.waitlist]), 
            patients, return_indices=True)[1])


    def _remove_dead_patients(self, policy) -> int:
        
        dead_patients_indices = np.where(np.array([p.time_to_live for p in self.waitlist]) <= 0)[0] # selects patient IDs when Sim_Patient.ttl <= 0
        tmp = self.waitlist[dead_patients_indices]
        dead_patients_ids = np.array([p.id for p in tmp])
        
        self._remove_patients(dead_patients_ids)                        # remove patients from self.waitlist
        policy.remove_x(dead_patients_ids)                              # remove patients from policy: policy.remove_x(list)
        
        ttl = self.DATA.iloc[dead_patients_indices].ttl.to_numpy()

        #self.stats.population_life_years += ttl.sum()
        return dead_patients_ids


    def _age_patients(self, days: int=1):
        self.waitlist = np.array([p.age(days) for p in self.waitlist])


    def _sample_patients(self) -> list:
        # returns a list of patient IDs
        n = np.abs(np.round(                                                # We sample an amount (n) from a normal distribution
            np.random.normal(loc=self.patient_count / self.days, size=1)    # this ensures the average of integers equals that of
        )).item()                                                           # patient_count / days. Note, the scale may be adjusted
        n = int(n)                                                          # should this be too extreme

        patients = self.patients.sample(n=n).index

        new_patients = np.array([
            Sim_Patient(
                id=patients[i],
                time_to_live=self.DATA.iloc[i].ttl) 
            for i in range(len(patients))
        ])
        
        self.waitlist = np.append(self.waitlist, new_patients)
        self.waitlist = np.unique(self.waitlist)

        return np.array(patients)


    def _sample_organs(self) -> list:
        # returns a list of organ IDs
        n = np.abs(np.round(
            np.random.normal(loc=(self.patient_count / self.days) * self.organ_deficit, size=1)
        )).item()
        n = int(n)

        organs = self.organs.sample(n=n).index
        return np.array(organs)

