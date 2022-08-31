import heapq
import re
from abc import ABC, abstractclassmethod
from collections import Counter
from dataclasses import dataclass, field
from operator import attrgetter, itemgetter
from typing import Any

import lifelines
import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.cluster import KMeans

from organsync.data.data_module import OrganDataModule
from organsync.models.inference import Inference
from organsync.models.linear import MELD as MELDscore
from organsync.models.linear import MELD3 as MELD3score
from organsync.models.linear import MELD_na as MELDnascore
from organsync.policies import Organ, Patient

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# POLICY DEFINITIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# LivSim Policies (MELD and MELD-na)


class Policy(ABC):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information
        data: str = "test",
        #   of the transplant system
    ) -> None:

        self.name = name
        self.waitlist = initial_waitlist
        self.dm = dm
        if data == "test":
            self.test = dm._test_processed  # perform on test set
        if data == "all":
            self.test = dm._all_processed  # perform on all data

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


class MELD(Policy):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information of the transplant system
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

        self._setup()

    def _get_x(self, organ: str) -> int:
        if len(self.waitlist) == 0:
            raise ValueError("empty waitlist")

        X = max(self.waitlist, key=attrgetter("covariates"))
        self.remove_x([X.id])
        return X.id

    def get_xs(self, organs: np.ndarray) -> np.ndarray:
        if len(organs) == 0 or len(self.waitlist) == 0:
            return np.array([])

        if len(organs) > len(self.waitlist):
            return np.array([self._get_x(organs[i]) for i in range(len(self.waitlist))])
        return np.array([self._get_x(organ) for organ in organs])

    def add_x(self, x: np.ndarray) -> None:
        if len(x) == 0:
            return

        MELD_score = self._meld(x)
        X = [Patient(id=x[i], covariates=MELD_score[i]) for i in range(len(x))]

        self.waitlist = np.append(self.waitlist, X)
        self.waitlist = np.unique(
            self.waitlist
        )  # make sure there no patient is containted twice

    def _setup(self) -> None:
        MELD_score = self._meld(self.waitlist)

        self.waitlist = np.array(
            [
                Patient(id=self.waitlist[i], covariates=MELD_score[i])
                for i in range(len(self.waitlist))
            ]
        )
        self.waitlist = np.unique(self.waitlist)

    def _meld(self, patients: np.ndarray) -> np.ndarray:
        # params -
        # :patients: array(int) - list of patient IDs

        ps = self.test.loc[self.test.index.isin(patients)].copy()
        ps.loc[:, self.dm.real_cols] = self.dm.scaler.inverse_transform(
            ps[self.dm.real_cols]
        )

        MELD_score = MELDscore().score(
            serum_bilirubin=ps.SERUM_BILIRUBIN,
            inr=ps.INR,
            serum_creatinine=ps.SERUM_CREATININE,
        )

        return MELD_score.to_numpy()

    def score(self, patients: np.ndarray) -> np.ndarray:
        return self._meld(patients)

    def remove_x(self, x: np.ndarray) -> None:
        for patient in x:
            self.waitlist = np.delete(
                self.waitlist,
                np.where(np.array([p.id for p in self.waitlist]) == patient)[0],
            )


class MELD_na(MELD):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information of the transplant system
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

    def _meld(self, patients: np.ndarray) -> np.ndarray:
        # We can simply inherit from MELD as the only part
        # that changes is they way we compute a MELD score
        # by adding the last term in MELD_score.

        ps = self.test.loc[self.test.index.isin(patients)].copy()
        ps.loc[:, self.dm.real_cols] = self.dm.scaler.inverse_transform(
            ps[self.dm.real_cols]
        )

        MELD_score = MELDnascore().score(
            serum_bilirubin=ps.SERUM_BILIRUBIN,
            inr=ps.INR,
            serum_creatinine=ps.SERUM_CREATININE,
            serum_sodium=ps.SERUM_SODIUM,
        )
        return MELD_score.to_numpy()


class MELD3(MELD):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information of the transplant system
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

    def _meld(self, patients: np.ndarray) -> np.ndarray:
        # We can simply inherit from MELD as the only part
        # that changes is they way we compute a MELD score
        # by adding the last term in MELD_score.

        ps = self.test.loc[self.test.index.isin(patients)].copy()
        ps.loc[:, self.dm.real_cols] = self.dm.scaler.inverse_transform(
            ps[self.dm.real_cols]
        )

        MELD_score = MELD3score().score(
            sex=ps.SEX.map({0: "M", 1: "F"}),
            serum_bilirubin=ps.SERUM_BILIRUBIN,
            inr=ps.INR,
            serum_creatinine=ps.SERUM_CREATININE,
            serum_sodium=ps.SERUM_SODIUM,
            serum_albumin=ps.SERUM_ALBUMIN,
        )
        return MELD_score.to_numpy()


# Naive FIFO policy
class FIFO(Policy):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information of the transplant system
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

    def remove_x(self, x: np.ndarray) -> None:
        for patient in x:
            self.waitlist = np.delete(
                self.waitlist, np.where(self.waitlist == patient)[0]
            )

    def add_x(self, x: np.ndarray) -> None:
        self.waitlist = np.append(self.waitlist, x)

    def get_xs(self, organs: np.ndarray) -> np.ndarray:
        patients = self.waitlist[: len(organs)]
        self.remove_x(patients)

        return patients


class MaxPolicy(Policy):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information of the transplant system
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

        self._setup()

    def _setup(self) -> None:
        self.x_cols = self.dm.x_cols
        waitlist_patients = self.test.loc[self.waitlist, self.x_cols].copy().to_numpy()

        self.waitlist = np.array(
            [
                Patient(id=self.waitlist[i], covariates=waitlist_patients[i])
                for i in range(len(self.waitlist))
            ]
        )
        self.waitlist = np.unique(self.waitlist)

    def get_xs(self, organs: np.ndarray) -> np.ndarray:
        if len(organs) == 0 or len(self.waitlist) == 0:
            return np.array([])

        if len(organs) > len(self.waitlist):
            for i in range(len(self.waitlist)):
                patient_ids = [
                    self._get_x(organs[i]) for i in range(len(self.waitlist))
                ]
                return patient_ids

        patient_ids = [self._get_x(organ) for organ in organs]

        return patient_ids

    def _get_x(self, organ: int) -> int:
        patient_covariates = np.array([p.covariates for p in self.waitlist])
        organ_covariates = self.test.loc[organ, self.dm.o_cols].to_numpy()

        scores = self._calculate_scores(patient_covariates, [organ_covariates])
        top_index = np.argmax(scores)
        patient_id = self.waitlist[top_index].id
        self.remove_x([patient_id])

        return patient_id

    def add_x(self, x: np.ndarray) -> None:
        if len(x) == 0:
            return

        patient_covariates = self.test.loc[x, self.x_cols].copy().to_numpy()
        patients = [
            Patient(id=x[i], covariates=patient_covariates[i]) for i in range(len(x))
        ]
        self.waitlist = np.append(self.waitlist, patients)
        self.waitlist = np.unique(self.waitlist)

    def remove_x(self, x: np.ndarray) -> None:
        for patient in x:
            self.waitlist = np.array([p for p in self.waitlist if p.id not in x])

    @abstractclassmethod
    def _calculate_scores(
        self, x_covariates: np.ndarray, o_covariates: np.ndarray
    ) -> np.ndarray:
        # this method should return, for each patient in
        # x_covariates, the score of that patient associated
        # with o_covariates. Note that o_covariates is just
        # one organ. This allows to remove the selected patient
        # from the waitlist.
        ...


class BestMatch(MaxPolicy):
    def __init__(
        self,
        name: str,
        initial_waitlist: np.ndarray,
        dm: OrganDataModule,
        inference: Inference,
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)
        self.inference = inference

    def _setup(self) -> None:
        self.x_cols = self.dm.x_cols  # [self.dm.x_cols != 'CENS']
        waitlist_patients = self.test.loc[self.waitlist, self.x_cols].copy().to_numpy()

        self.waitlist = np.array(
            [
                Patient(id=self.waitlist[i], covariates=waitlist_patients[i])
                for i in range(len(self.waitlist))
            ]
        )
        self.waitlist = np.unique(self.waitlist)

    def _calculate_scores(
        self, x_covariates: np.ndarray, o_covariates: np.ndarray
    ) -> np.ndarray:
        return [
            self.inference(np.array([patient]), o_covariates)
            for patient in x_covariates
        ]


class SickestFirst(MaxPolicy):
    def __init__(
        self,
        name: str,
        initial_waitlist: np.ndarray,
        dm: OrganDataModule,
        inference: Inference,
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)
        self.inference = inference

    def _setup(self) -> None:
        self.x_cols = self.dm.x_cols  # [self.dm.x_cols != 'CENS']
        waitlist_patients = self.test.loc[self.waitlist, self.x_cols].copy().to_numpy()

        self.waitlist = np.array(
            [
                Patient(id=self.waitlist[i], covariates=waitlist_patients[i])
                for i in range(len(self.waitlist))
            ]
        )
        self.waitlist = np.unique(self.waitlist)

    def _calculate_scores(
        self, x_covariates: np.ndarray, o_covariates: np.ndarray
    ) -> np.ndarray:
        return [self.inference(np.array([patient])) ** -1 for patient in x_covariates]


# Contemporary UK policy
class TransplantBenefit(MaxPolicy):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information of the transplant system
        inference: Inference,
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

        self.inference = inference

    def _setup(self) -> None:
        self.x_cols = self.dm.x_cols[self.dm.x_cols != "CENS"]
        waitlist_patients = self.test.loc[self.waitlist, self.x_cols].copy().to_numpy()

        self.waitlist = np.array(
            [
                Patient(id=self.waitlist[i], covariates=waitlist_patients[i])
                for i in range(len(self.waitlist))
            ]
        )
        self.waitlist = np.unique(self.waitlist)

    def _calculate_scores(
        self, x_covariates: np.ndarray, o_covariates: np.ndarray
    ) -> np.ndarray:
        return [
            self.inference(np.array([patient]), o_covariates)
            for patient in x_covariates
        ]


class TransplantBenefit_original(MaxPolicy):
    def __init__(
        self,
        name: str,
        initial_waitlist: np.ndarray,
        dm: OrganDataModule,
        inference: Inference,
        data: str = "test",
        model: str = "tbs",  # can also be 'm1' or 'm2'
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)
        self.inference = inference
        model_indices = {"tbs": 0, "m1": 1, "m2": 2}
        self.model_index = model_indices[model]

    def _setup(self) -> None:
        self.x_cols = self.dm.x_cols[self.dm.x_cols != "CENS"]
        waitlist_patients = self.test.loc[self.waitlist, self.x_cols].copy().to_numpy()

        self.waitlist = np.array(
            [
                Patient(id=self.waitlist[i], covariates=waitlist_patients[i])
                for i in range(len(self.waitlist))
            ]
        )
        self.waitlist = np.unique(self.waitlist)

    def _calculate_scores(
        self, x_covariates: np.ndarray, o_covariates: np.ndarray
    ) -> np.ndarray:
        o_covariates = np.tile(o_covariates, reps=(x_covariates.shape[0], 1))

        DATA = pd.DataFrame(
            data=np.append(x_covariates, o_covariates, axis=1),
            columns=[*self.x_cols, *self.dm.o_cols],
        )

        # patient one-hots
        ohe = [
            "DCOD",
            "PATIENT_LOCATION",
            "RASCITES",
            "RENAL_SUPPORT",
            "RREN_SUP",
            "SEX",
        ]
        for variable in ohe:
            VAR = DATA.filter(regex=variable)
            VAR = pd.get_dummies(VAR).idxmax(1).values
            VAR = [int(re.findall("\\d+", V)[0]) for V in VAR]
            DATA.loc[:, variable] = VAR

        power = -1 if self.model_index == 1 else 1  # turn max into min policy for M1

        return self.inference.infer(x=DATA)[self.model_index] ** power


# ML-based policies
#   NOTE: ConfidentMatch and TransplantBenefit are essentially
#       the same policy, where we maximise some inferred value.
#       Also MELD and MELD_na fall in this category, though the
#       inference is not stochastic
class ConfidentMatch(MaxPolicy):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information of the transplant system
        inference: Inference,
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

        self.inference = inference

    def _calculate_scores(
        self, x_covariates: np.ndarray, o_covariates: np.ndarray
    ) -> np.ndarray:
        return [
            self.inference(np.array([patient]), o_covariates)
            for patient in x_covariates
        ]


class OrganITE(MaxPolicy):
    def __init__(
        self,
        name: str,
        initial_waitlist: np.ndarray,
        dm: OrganDataModule,
        inference_ITE: Inference,
        inference_VAE: Inference,
        a: float = 1.0,
        b: float = 1.0,
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)
        self.inference_ITE = inference_ITE
        self.inference_VAE = inference_VAE

        self.a = a
        self.b = b

    def _setup(self) -> None:
        super()._setup()

        # self.k_means = self.inference_ITE.model.cluster                                 # LOAD CLUSTERS FROM inference_ITE

    def _calculate_scores(
        self, x_covariates: np.ndarray, o_covariates: np.ndarray
    ) -> np.ndarray:
        scores = [
            self._calculate_score(
                np.array([patient]), np.array(o_covariates, dtype=float)
            )
            for patient in x_covariates
        ]

        return scores

    def _calculate_score(self, patient: np.ndarray, organ: np.ndarray) -> np.ndarray:
        ITE = self.inference_ITE(patient, organ)

        ITE *= self._get_lambda(patient, organ)

        return ITE

    def _get_optimal_organ(self, patient: np.ndarray) -> np.ndarray:
        sample_organs = self.dm._train_processed.sample(n=512)[
            self.dm.o_cols
        ].to_numpy()
        repeated_patients = np.repeat(patient, 512, axis=0)
        ITEs = self.inference_ITE(repeated_patients, sample_organs)
        optimal_organ_ix = np.argmax(ITEs)
        optimal_organ = sample_organs[optimal_organ_ix]

        return optimal_organ.reshape(1, -1)

    def _get_lambda(self, patient: np.ndarray, organ: np.ndarray) -> np.ndarray:
        optimal_organ = self._get_optimal_organ(patient)
        propensity = self._get_propensities([optimal_organ])
        distance = self._get_distances(optimal_organ, organ)

        lam = ((propensity + 0.000001) ** (-self.a)) * (
            distance + 0.000001 ** (-self.b)
        )
        return lam

    def _get_distances(self, organ_A: np.ndarray, organ_B: np.ndarray) -> np.ndarray:
        distance = scipy.spatial.distance.euclidean(organ_A, organ_B)
        return distance

    def _get_ITE(self, organ: np.ndarray) -> np.ndarray:
        patients = np.array([p.covariates for p in self.waitlist])
        organs = np.repeat(organ, len(patients), axis=0)
        null_organs = np.zeros(organs.shape)

        Y_1 = self.inference_ITE(patients, organs)
        Y_0 = self.inference_ITE(patients, null_organs)

        return (Y_1 - Y_0).numpy()

    def _get_propensities(
        self,
        o_covariates: np.ndarray,
    ) -> np.ndarray:
        return self.inference_VAE(torch.Tensor(o_covariates).double())

    def _get_patients(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        return self._get_instances(x, self.dm.x_cols, data_class=Patient, train=train)

    def _get_organs(self, o: np.ndarray, train: bool = False) -> np.ndarray:
        return self._get_instances(o, self.dm.o_cols, data_class=Organ, train=train)

    def _get_instances(
        self,
        l: np.ndarray,
        cols: np.ndarray,
        data_class: Any,
        train: bool = False,
    ) -> np.ndarray:
        data = self.test
        if train:
            data = self.dm._train_processed
        covariates = data.loc[data.index.isin(l), cols].copy()
        types = np.array(
            [
                data_class(id=l[i], covariates=covariates.iloc[i].to_numpy())
                for i in range(len(l))
            ]
        )

        return types


class OrganSyncMax(MaxPolicy):
    def __init__(
        self,
        name: str,
        initial_waitlist: np.ndarray,
        dm: OrganDataModule,
        inference_0: Inference,
        inference_1: Inference,
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

        self.inference_0 = inference_0
        self.inference_1 = inference_1

    def _calculate_scores(
        self, x_covariates: np.ndarray, o_covariates: np.ndarray
    ) -> np.ndarray:
        assert len(o_covariates) == 1, "_calculate_scores expects a single organ"
        o_covariates = torch.Tensor(o_covariates).double()
        o_covariates = o_covariates.repeat((len(x_covariates), 1))

        y_1, _, _ = self.inference_1(x_covariates, o_covariates.numpy(), SC=True)
        y_0, _, _ = self.inference_1(
            x_covariates, np.zeros(o_covariates.shape), SC=True
        )

        return y_1 - y_0


@dataclass(order=True)
class PrioritizedPatient:
    priority: float
    item: Any = field(compare=False)

    def __str__(self) -> str:
        return f"priority: {self.priority}\nID: {self.item.id}"


class OrganSync(Policy):
    def __init__(
        self,
        name: str,
        initial_waitlist: np.ndarray,
        dm: OrganDataModule,
        K: int,
        inference_0: Inference,
        inference_1: Inference,
        max_contributors: int = 30,
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

        self.K = K  # amount of queues (hyperparameter)
        self.max_contributors = (
            max_contributors  # max_contributors is to limit to the top
        )
        # max_contributors of the LASSO'd vector

        # OrganSync uses 2 models:
        self.inference_0 = inference_0  #   - infrence_0 for no organ predictions
        self.inference_1 = inference_1  #   - inference_1 for with organ predictions

        self._setup()

    def _setup(self) -> None:

        organs = self.dm._train_processed[
            self.dm.o_cols
        ]  # get organ data from training set
        self.k_means = KMeans(n_clusters=self.K)  # build K organ clusters on dm
        self.k_means.fit(organs)

        self.queues: dict = {
            i: [] for i in range(self.K)
        }  # setup initial waitlist (multiple queue's)
        for (
            v
        ) in (
            self.queues.values()
        ):  # transform lists to heapq's; note that heapq.heapify
            heapq.heapify(v)  # has not return statement, requiring a loop as this

        inferred_optimal_queues = (
            self._get_optimal_organ(  # we sample 5000 patients from the training set
                x=np.array(
                    self.dm._train_processed.sample(n=5000).index
                ),  # and calculate what their ideal organ would be
                train=True,
            )
        )  # we than use these amounts to compute an estimate
        amount: Counter = Counter(
            np.array(inferred_optimal_queues).flatten()
        )  # for the incoming rate into each queue

        self.queue_rates = {
            i: amount[i] / 5000 for i in amount.keys()
        }  # each queue gets assigned its incoming rate

        self.add_x(self.waitlist)

    def remove_x(self, x: np.ndarray) -> None:
        for queue in self.queues.keys():
            for patient in x:
                self.queues[queue] = [
                    p for p in self.queues[queue] if p.item.id not in x
                ]
                heapq.heapify(self.queues[queue])

    def get_xs(self, organs: np.ndarray) -> np.ndarray:
        if len(organs) == 0:
            return np.array([])

        organs = self._get_organs(organs)
        organ_clusters = self.k_means.predict(
            [o.covariates for o in organs]
        )  # predict cluster for organs

        selected_patients = np.array([])
        for organ_cluster in organ_clusters:
            amount_of_patients_on_queues = {
                q: len(self.queues[q]) for q in self.queues.keys()
            }
            total_amount = np.array(list(amount_of_patients_on_queues.values())).sum()

            if total_amount == 0:  # should there be more organs incoming then
                return selected_patients  # there are patients on the queue, break
                # preemtively

            try:
                patient = heapq.heappop(
                    self.queues[organ_cluster]
                )  # pop first in line of corresponding cluster
            except BaseException:  # should there be no patient on the queue of interest
                longest_queue = max(  # we select the longest queue, refraining the system
                    amount_of_patients_on_queues.items(),  # from getting overcrowded
                    key=itemgetter(1),
                )[
                    0
                ]

                patient = heapq.heappop(self.queues[longest_queue])

            selected_patients = np.append(selected_patients, patient.item.id)

        return selected_patients

    def add_x(self, x: np.ndarray) -> None:
        if len(x) == 0:
            return
        patients = self._get_patients(x)

        optimal_organ_clusters = self._get_optimal_organ(
            x
        )  # gather optimal organs for x
        priorities = self._get_priorities(x)  # gather hazard for x
        wait_times = self._get_waittimes(
            priorities
        )  # assign x to corresponding queue with priority = hazard

        prioritized_patients = [
            PrioritizedPatient(
                priority=priorities[i],  # make prioritized list of patients
                item=patients[i],
            )
            for i in range(len(patients))
        ]

        for i, prioritized_patient in enumerate(prioritized_patients):
            assigned = False
            for j in optimal_organ_clusters[i]:
                if prioritized_patient.priority > wait_times[i, j] and not assigned:
                    heapq.heappush(self.queues[j], prioritized_patient)
                    assigned = True

    def _get_waittimes(self, priorities: np.ndarray) -> np.ndarray:
        # from priorities, deliver K waittimes (len(priorities, k))
        wait_times = np.ndarray((0, self.K))
        for priority in priorities:
            wait_time = [
                [
                    np.count_nonzero(
                        np.where(
                            np.array(self.queues[k])
                            < PrioritizedPatient(priority=priority, item=None),
                            self.queues[k],
                            np.zeros(len(self.queues[k])),
                        )
                    )
                    / self.queue_rates[k]  # Little's law
                    for k in range(self.K)
                ]
            ]
            wait_times = np.append(wait_times, wait_time, axis=0)

        return wait_times

    def _get_priorities(self, x: np.ndarray) -> np.ndarray:
        # as the hazard is only used for queue priority, we
        # are only concerend for survival without organs
        # as such, we will only use inference_0 for this.

        patients = self._get_patients(x)
        patient_covs = [p.covariates for p in patients]
        y, a, ixs = self.inference_0(
            np.array(patient_covs), o=np.empty((len(x), 0)), SC=True
        )  # gather estimated outcomes and contributing
        a = np.array([*a])  # indices using synthetic control on infrence_0
        ixs = np.repeat(ixs.numpy().reshape(1, -1), len(x), axis=0)

        contributors = a.argsort(axis=1)[
            :, : self.max_contributors
        ]  # make sparse vector
        dm_0 = (
            self.inference_0.model.trainer.datamodule
        )  # we are inferring using the control of
        # our training data

        priorities = []
        for c in contributors:
            sparse_train_set = dm_0._train_processed.iloc[c]
            kmf = lifelines.KaplanMeierFitter()  # Kaplain-meier has no expected time to
            kmf.fit(  # event in lifelines. Instead we take the
                sparse_train_set.Y * dm_0.std
                + dm_0.mean,  # median, i.e., before the median, 50% died
                event_observed=sparse_train_set.CENS,
            )

            priority = (
                kmf.median_survival_time_
            )  # note that in the paper, those with highest
            priorities.append(priority)  # priority go first. this is the same here
            # as python uses a min-heap as priority queue
            # as such, we take the median survival time, if
            # a more standard max-heap is used, we recommend
            # the expected hazard (using, e.g. a Nelson-Aalen
            # estimate), like described in the paper

        return priorities

    def _get_optimal_organ(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        patients = self._get_patients(x, train=train)
        optimal_organ_clusters = []

        cluster_centers = (
            self.k_means.cluster_centers_
        )  # instead of iterating over each organ in the training
        for (
            patient
        ) in patients:  # set, we iterate over the K cluster centers. This makes
            patient = patient.covariates.reshape(
                1, -1
            )  # no difference in our implementation vs. the algorithm
            patient_copies = np.repeat(
                patient, self.K, axis=0
            )  # described in our paper, as the algorithm in our paper
            organs = np.array(
                cluster_centers
            )  # would later compare the distance to the closest center
            y = self.inference_1(patient_copies, o=organs)[
                0
            ]  # to assign a cluster anyway.
            optimal_organ_clusters.append(np.argsort(-y.numpy().flatten()))

        return optimal_organ_clusters

    def _get_patients(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        return self._get_instances(x, self.dm.x_cols, data_class=Patient, train=train)

    def _get_organs(self, o: np.ndarray, train: bool = False) -> np.ndarray:
        return self._get_instances(o, self.dm.o_cols, data_class=Organ, train=train)

    def _get_instances(
        self,
        l: np.ndarray,
        cols: np.ndarray,
        data_class: Any,
        train: bool = False,
    ) -> np.ndarray:
        data = self.test
        if train:
            data = self.dm._train_processed
        covariates = data.loc[data.index.isin(l), cols].copy()
        types = np.array(
            [
                data_class(id=l[i], covariates=covariates.iloc[i].to_numpy())
                for i in range(len(l))
            ]
        )

        return types
