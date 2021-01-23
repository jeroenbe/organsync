import heapq
import numpy as np
import pandas as pd
import lifelines

from sklearn.cluster import KMeans

from abc import ABC
from abc import abstractclassmethod

from dataclasses import dataclass, field
from typing import Any
from collections import Counter

from src.data.data_module import OrganDataModule


from operator import attrgetter, itemgetter

@dataclass
class Organ:
    id: int
    covariates: np.array

    def __eq__(self, other):
        if not isinstance(other, Organ):
            return NotImplemented
        return self.id == other.id
    
    def __gt__(self, other):
        if not isinstance(other, Organ):
            return NotImplemented
        return self.id > other.id

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
    def get_xs(self, organs: list) -> int:
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
    def add_x(self, x: list):
        # Whenever a patient enters the transplant system
        # add_x is called. It is the policies task to maintain
        # system state.
        #
        # params - 
        # :x: int - patient ID, for reference to dm.ID (note,
        #   dm.ID covers patient-organ pair)
        pass
    
    @abstractclassmethod
    def remove_x(self, x: list):
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
        if len(self.waitlist) == 0:
            return

        X = max(self.waitlist, key=attrgetter('covariates'))
        self.remove_x([X.id])
        return X.id
    
    def get_xs(self, organs):
        if len(organs) == 0:
            return np.array([])
        return np.array([self._get_x(organ) for organ in organs])

    def add_x(self, x):
        if len(x) == 0:
            return

        MELD_score = self._meld(x)
        X = [Patient(id=x[i], covariates=MELD_score[i]) for i in range(len(x))]

        self.waitlist = np.append(self.waitlist, X)
        self.waitlist = np.unique(self.waitlist)        # make sure there no patient is containted twice
    
    def _setup(self):
        MELD_score = self._meld(self.waitlist)

        self.waitlist = np.array([Patient(id=self.waitlist[i], covariates=MELD_score[i]) for i in range(len(self.waitlist))])
        self.waitlist = np.unique(self.waitlist)

    def _meld(self, patients):
        # params - 
        # :patients: array(int) - list of patient IDs

        ps = self.test.loc[self.test.index.isin(patients)].copy()
        ps.loc[:, self.dm.real_cols] = self.dm.scaler.inverse_transform(ps[self.dm.real_cols])

        # DEFINITION OF (standard) MELD: https://en.wikipedia.org/wiki/Model_for_End-Stage_Liver_Disease#Determination
        MELD_score = 3.79 * np.log(ps.SERUM_BILIRUBIN) + 11.2 * np.log(ps.INR) + 9.57 * np.log(ps.SERUM_CREATININE) + 6.43
        return  MELD_score.to_numpy()

    def remove_x(self, x):
        self.waitlist = np.delete(self.waitlist, np.where(np.array([p.id for p in self.waitlist]) == x)[0])

class MELD_na(MELD):
    def __init__(self, name, initial_waitlist, dm):
        super().__init__(name, initial_waitlist, dm)
    
    def _meld(self, patients):
        # We can simply inherit from MELD as the only part
        # that changes is they way we compute a MELD score 
        # by adding the last term in MELD_score.

        ps = self.test.loc[self.test.index.isin(patients)].copy()
        ps.loc[:, self.dm.real_cols] = self.dm.scaler.inverse_transform(ps[self.dm.real_cols])
        
        # MELD-na: MELD + 1.59*(135-SODIUM(mmol/l)) (https://github.com/kartoun/meld-plus/raw/master/MELD_Plus_Calculator.xlsx)
        MELD_score = super()._meld(patients) + 1.59 * (135 - ps.SERUM_SODIUM)
        return MELD_score.to_numpy()

# Naive FIFO policy
class FIFO(Policy):
    def __init__(self, name, initial_waitlist, dm):
        super().__init__(name, initial_waitlist, dm)
    
    def remove_x(self, x: list):
        self.waitlist = np.delete(self.waitlist, np.where(self.waitlist == x)[0])
    
    def add_x(self, x: list):
        self.waitlist = np.append(self.waitlist, x)
    
    def get_xs(self, organs: list):
        patients = self.waitlist[:len(organs)]
        self.remove_x(patients)

        return patients


# Contemporary UK policy
class TransplantBenefit(Policy):
    def __init__(self, name, initial_waitlist, dm):
        super().__init__(name, initial_waitlist, dm)

# ML-based policies
#   NOTE: ConfidentMatch and TransplantBenefit are essentially
#       the same policy, where we maximise some inferred value.
#       Also MELD and MELD_na fall in this category, though the
#       inference is not stochastic
class ConfidentMatch(Policy):
    def __init__(self, name, initial_waitlist, dm):
        super().__init__(name, initial_waitlist, dm)


class OrganITE(Policy):
    def __init__(self, name, initial_waitlist, dm):
        super().__init__(name, initial_waitlist, dm)


@dataclass(order=True)
class PrioritizedPatient:
    priority: float
    item: Any=field(compare=False)

    def __str__(self):
        return f'priority: {self.priority}\nID: {self.item.id}'


class OrganSync(Policy):
    def __init__(self, name, initial_waitlist, dm, K, inference_0, inference_1, max_contributors: int=30):
        super().__init__(name, initial_waitlist, dm)

        self.K = K                                                                      # amount of queues (hyperparameter)
        self.max_contributors = max_contributors                                        # max_contributors is to limit to the top
                                                                                        # max_contributors of the LASSO'd vector

                                                                                        # OrganSync uses 2 models:
        self.inference_0 = inference_0                                                  #   - infrence_0 for no organ predictions
        self.inference_1 = inference_1                                                  #   - inference_1 for with organ predictions

        self._setup()
    
    def _setup(self):

        organs = self.dm._train_processed[self.dm.o_cols]                               # get organ data from training set
        self.k_means = KMeans(n_clusters=self.K)                                        # build K organ clusters on dm
        self.k_means.fit(organs)

        self.queues = {i: [] for i in range(self.K)}                                    # setup initial waitlist (multiple queue's)
        for v in self.queues.values():                                                  # transform lists to heapq's; note that heapq.heapify
            heapq.heapify(v)                                                            # has not return statement, requiring a loop as this

        inferred_optimal_queues = self._get_optimal_organ(                              # we sample 5000 patients from the training set
            x=np.array(self.dm._train_processed.sample(n=5000).index),                  # and calculate what their ideal organ would be
            train=True)                                                                 # we than use these amounts to compute an estimate
        amount = Counter(np.array(inferred_optimal_queues).flatten())                   # for the incoming rate into each queue

        self.queue_rates = {i: amount[i] / 500 for i in amount.keys()}                  # each queue gets assigned its incoming rate

        self.add_x(self.waitlist)

    def remove_x(self, x: list):
        for queue in self.queues.keys():
            for patient in x:
                self.queues[queue] = [p for p in self.queues[queue] if p.item.id not in x]
                heapq.heapify(self.queues[queue])

    
    def get_xs(self, organs: int):
        if len(organs) == 0:
            return np.array([])

        organs = self._get_organs(organs)       
        organ_clusters = self.k_means.predict([o.covariates for o in organs])           # predict cluster for organs
        
        selected_patients = np.array([])
        for organ_cluster in organ_clusters:
            amount_of_patients_on_queues = {
                q: len(self.queues[q]) for q in self.queues.keys()} 
            total_amount = np.array(list(amount_of_patients_on_queues.values())).sum()
            
            if total_amount == 0:                                                       # should there be more organs incoming then
                return selected_patients                                                # there are patients on the queue, break 
                                                                                        # preemtively
            
            try:
                patient = heapq.heappop(self.queues[organ_cluster])                     # pop first in line of corresponding cluster
            except:                                                                     # should there be no patient on the queue of interest
                longest_queue = max(                                                    # we select the longest queue, refraining the system
                    amount_of_patients_on_queues.items(),                               # from getting overcrowded
                    key=itemgetter(1))[0]
                
                patient = heapq.heappop(self.queues[longest_queue])

            selected_patients = np.append(selected_patients, patient.item.id)


        return selected_patients

    def add_x(self, x: list):
        if len(x) == 0:
            return
        patients = self._get_patients(x)
        
        optimal_organ_clusters = self._get_optimal_organ(x)                             # gather optimal organs for x
        priorities = self._get_priorities(x)                                            # gather hazard for x
        wait_times = self._get_waittimes(priorities)                                    # assign x to corresponding queue with priority = hazard

        prioritized_patients = [PrioritizedPatient(priority=priorities[i],              # make prioritized list of patients
            item=patients[i]) for i in range(len(patients))]
        
        for i, prioritized_patient in enumerate(prioritized_patients):
            assigned=False
            for j in optimal_organ_clusters[i]:
                if not assigned and prioritized_patient.priority > wait_times[i, j]:
                    heapq.heappush(self.queues[j], prioritized_patient)
                    assigned=True

    def _get_waittimes(self, priorities: list):
        # from priorities, deliver K waittimes (len(priorities, k))
        wait_times = np.ndarray((0, self.K))
        for priority in priorities:
            wait_time = [[np.count_nonzero(
                np.where(np.array(self.queues[k]) < PrioritizedPatient(priority=priority, item=None), 
                self.queues[k], np.zeros(len(self.queues[k])))) / self.queue_rates[k]   # Little's law
                for k in range(self.K)]]
            wait_times = np.append(wait_times, wait_time, axis=0)

        return wait_times

    def _get_priorities(self, x: list):
        # as the hazard is only used for queue priority, we 
        # are only concerend for survival without organs
        # as such, we will only use inference_0 for this.

        patients = self._get_patients(x)
        patient_covs = [p.covariates for p in patients]
        y, a, ixs = self.inference_0(
            np.array(patient_covs), o=np.empty((len(x), 0)), SC=True)                   # gather estimated outcomes and contributing
        a = np.array([*a])                                                              # indices using synthetic control on infrence_0
        ixs = np.repeat(ixs.numpy().reshape(1, -1), len(x), axis=0)

        contributors = a.argsort(axis=1)[:,:self.max_contributors]                      # make sparse vector
        dm_0 = self.inference_0.model.trainer.datamodule                                # we are inferring using the control of
                                                                                        # our training data
        
        priorities = []
        for c in contributors:
            sparse_train_set = dm_0._train_processed.iloc[c]
            naf = lifelines.NelsonAalenFitter()                                         # in order to calculate the hazard of 
            naf.fit(                                                                    # a kaplan-meier survival function one
                sparse_train_set.Y * dm_0.std + dm_0.mean,                              # has to use a nelson-aalen fitter
                event_observed=sparse_train_set.CENS)

            priority = naf.cumulative_hazard_.sum().item()                              # note that in the paper, those with lowest
                                                                                        # priority go first. this is the same here
                                                                                        # as python uses a min-heap as priority queue
            priorities.append(priority)

        return priorities

    def _get_optimal_organ(self, x: list, train: bool=False):
        patients = self._get_patients(x, train=train)
        optimal_organ_clusters = []

        cluster_centers = self.k_means.cluster_centers_                                 # instead of iterating over each organ in the training
        for patient in patients:                                                        # set, we iterate over the K cluster centers. This makes
            patient = patient.covariates.reshape(1, -1)                                 # no difference in our implementation vs. the algorithm
            patient_copies = np.repeat(patient, self.K, axis=0)                         # described in our paper, as the algorithm in our paper
            organs = np.array(cluster_centers)                                          # would later compare the distance to the closest center
            y = self.inference_1(patient_copies, o=organs)[0]                           # to assign a cluster anyway.
            optimal_organ_clusters.append(np.argsort(-y.numpy().flatten()))
                                                                
        return optimal_organ_clusters

    def _get_patients(self, x: list, train: bool=False):
        return self._get_instances(x, self.dm.x_cols, data_class=Patient, train=train)
    
    def _get_organs(self, o: list, train: bool=False):
        return self._get_instances(o, self.dm.o_cols, data_class=Organ, train=train)

    def _get_instances(self, l: list, cols: list, data_class: dataclass, train: bool=False):
        data = self.test
        if train:
            data = self.dm._train_processed
        covariates = data.loc[data.index.isin(l), cols].copy()
        types = np.array([data_class(id=l[i], covariates=covariates.iloc[i].to_numpy()) for i in range(len(l))])

        return types