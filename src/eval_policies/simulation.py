import numpy as np
import pandas as pd

import torch

from dataclasses import dataclass, field

from abc import ABC, abstractclassmethod

from src.data.data_module import OrganDataModule
from src.models.organsync import OrganSync_Network

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
    patients_transplanted: dict=field(default_factory=dict) # should be a dict of day: np.array w shape (2, n_patients_on_day)
    organs_transplanted: dict=field(default_factory=dict) 
    patients_died: dict=field(default_factory=dict)         # dict of day: np.array

    def __str__(self):
        return f'Deaths: {self.deaths}\nPopulation life-years: {self.population_life_years}\nTransplant count: {self.transplant_count}'
        

class Inference(ABC):
    def __init__(self, model, mean, std):
        self.model = model
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        return self.infer(x)
    
    @abstractclassmethod
    def infer(self, x):
        pass

class Inference_OrganSync(Inference):
    def __init__(self, model: OrganSync_Network, mean, std, SC: bool=False):
        super().__init__(model, mean, std)

        self.SC = SC
    
    def infer(self, x, o=None):
        with torch.no_grad():
            x = torch.Tensor(x).double()
            
            if self.SC and o is not None:
                o = torch.Tensor(o).double()
                _, _, y, synth_y = self.model.synthetic_control(x, o)
            else:
                y = self.model(x)
                y = y * self.std + self.mean
            return y


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
        self.DATA = dm._test_processed.copy(deep=True)  # These are the 'ground truth' datasets for
        self.patients = self.DATA[dm.x_cols]            # for patients and organs. From these we'll
        self.organs = self.DATA[dm.o_cols]              # create a fictive event calender.

        self.organ_deficit = organ_deficit      # For every 1 patient, there will be 
                                                # organ_deficit organs (usually less than 1)
        self.days = days                        # Amount of days the simulation runs (1 year standard)
        self.patient_count = patient_count      # Max amount of patients, on average this means 
                                                # patient_count/days amount per day.


        self.inference_0 = inference_0          # These models will function as ground truth
        self.inference_1 = inference_1          # outcomes. Note that LSAM and LivSim use simple
                                                # linear models. With an Inference class, this 
                                                # can be anything.

        self.waitlist = np.array([])                        # Get's initialized in self._setup. This allows
        self.initial_waitlist_size = initial_waitlist_size  # multiple runs with one Sim object.

        X_tmp = torch.Tensor(self.DATA[self.dm.x_cols].to_numpy())      # Add time to live (ttl) column
        self.DATA.loc[:,'ttl'] = self.inference_0(X_tmp).numpy()        # to DATA using inference_0


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

    def simulate(self, policy) -> Stats:
        # while not stop_critierum
        #   self.iterate(policy)

        for day in range(self.days):
            self.iterate(policy, day)


        return self.stats


    def _update_stats(self):
        pass
    

    def iterate(self, policy, _day):
        
        dead_patients = self._remove_dead_patients(policy)  # remove dead patients from waitlist (also in policy)
        amount_died = len(dead_patients)

        patients = self._sample_patients()                  # sample patient(s)
        organs = self._sample_organs()                      # sample organ(s)
        policy.add_x(patients)                              # add patient(s) and organ(s) to 
        transplant_patients = policy.get_xs(organs)         # the policy's internal waitlist
                                                            # and assign organs to patients

        
        #  CALCULATE TTL FOR transplant_patients BY policy
        organs_cov = self.organs.loc[organs].to_numpy()        # sample organ covariates for sampled indices
        patients_cov = self.patients.loc[                      # sample patient covariates for transplanted indices
            transplant_patients].to_numpy() 
        
        catted = np.append(patients_cov, organs_cov, axis=1)    # calculate time to live (ttl) with organ using inference_1
        ttl = np.array([self.inference_1(x) for x in catted])

        self._remove_patients(transplant_patients)              # remove transplanted patients from waitlist
                                                                # note that transplant_patients are automatically
                                                                # removed from the policy's internal waitlist
        
        
        self.stats.deaths += amount_died                        # update statistics
        self.stats.population_life_years += ttl.sum()
        self.stats.transplant_count += len(transplant_patients)
        
        self.stats.patients_died[f'{_day}'] = dead_patients
        self.stats.patients_transplanted[f'{_day}'] = transplant_patients
        self.stats.organs_transplanted[f'{_day}'] = organs


        self._age_patients()


    def _remove_patients(self, patients):
        self.waitlist = np.delete(self.waitlist, 
        np.intersect1d(
            np.array([p.id for p in self.waitlist]), 
            patients, return_indices=True)[1])


    def _remove_dead_patients(self, policy) -> int:
        
        dead_patients_indices = np.where(np.array([p.time_to_live for p in self.waitlist]) <= 0)[0] # selects patient IDs when Sim_Patient.ttl <= 0
        tmp = self.waitlist[dead_patients_indices]
        dead_patients_ids = np.array([p.id for p in tmp])
        
        self._remove_patients(dead_patients_ids)            # remove patients from self.waitlist
        policy.remove_x(dead_patients_ids)                  # remove patients from policy: policy.remove_x(list)
        
        return dead_patients_ids


    def _age_patients(self, days: int=1):
        self.waitlist = np.array([p.age(days) for p in self.waitlist])


    def _sample_patients(self) -> list:
        # returns a list of patient IDs
        n = np.abs(np.round(                                                # We sample an amount (n) from a normal distribution
            np.random.normal(loc=self.patient_count / self.days, size=1)    # this ensures the average of integers equals that of
        )).item()                                                           # patient_count / days. Note, the scale may be adjusted
        n = int(n)                                                          # should this be too extreme

        indices = np.random.randint(0, len(self.patients), (n, ))
        patients = self.patients.iloc[indices].index

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

        indices = np.random.randint(0, len(self.organs), (n, ))
        organs = self.organs.iloc[indices].index
        return np.array(organs)



