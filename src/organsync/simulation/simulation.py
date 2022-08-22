from dataclasses import dataclass, field
from datetime import date
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from dateutil.rrule import DAILY, rrule
from tqdm import tqdm

from organsync.data.data_module import OrganDataModule
from organsync.models.inference import Inference
from organsync.policies import Policy


@dataclass
class Sim_Patient:
    id: int
    time_to_live: int

    def age(self, days: int) -> "Sim_Patient":
        self.time_to_live -= days
        return self

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Sim_Patient):
            return NotImplemented
        return self.id == other.id

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Sim_Patient):
            return NotImplemented
        return self.id > other.id


@dataclass
class Stats:
    deaths: int = 0
    patients_seen: int = 0
    population_life_years: float = 0
    transplant_count: int = 0
    first_empty_day: int = -1
    patients_transplanted: dict = field(
        default_factory=dict
    )  # should be a dict of day: np.array w shape (2, n_patients_on_day)
    organs_transplanted: dict = field(default_factory=dict)
    patients_died: dict = field(default_factory=dict)  # dict of day: np.array

    def __str__(self) -> str:
        return f"Deaths: {self.deaths}\nPopulation life-years: {self.population_life_years}\nTransplant count: {self.transplant_count}\nFirst empty day: {self.first_empty_day}"


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


class Sim:
    def __init__(
        self,
        dm: OrganDataModule,
        initial_waitlist_size: int,
        inference_0: Inference,
        inference_1: Inference,
        organ_deficit: float,
        patient_count: int = 1000,
        days: int = 365,
    ) -> None:

        self.dm = dm
        self.DATA = dm._test_processed.copy(
            deep=True
        )  # These are the 'ground truth' datasets for
        self.patients = self.DATA[
            dm.x_cols
        ]  # for patients and organs. From these we'll
        self.organs = self.DATA[dm.o_cols]  # create a fictive event calender.

        self.organ_deficit = organ_deficit  # For every 1 patient, there will be
        # organ_deficit organs (usually less than 1)
        self.days = days  # Amount of days the simulation runs (1 year standard)
        self.patient_count = (
            patient_count  # Max amount of patients, on average this means
        )
        # patient_count/days amount per day.

        self.inference_0 = inference_0  # These models will function as ground truth
        self.inference_1 = inference_1  # outcomes. Note that LSAM and LivSim use simple
        # linear models. With an Inference class, this
        # can be anything.

        self.waitlist = np.array([])  # Get's initialized in self._setup. This allows
        self.initial_waitlist_size = (
            initial_waitlist_size  # multiple runs with one Sim object.
        )

        X_tmp = torch.Tensor(
            self.DATA[self.dm.x_cols].to_numpy()
        )  # Add time to live (ttl) column
        self.DATA.loc[:, "ttl"] = self.inference_0(X_tmp)[
            0
        ].numpy()  # to DATA using inference_0

        self.log_df = pd.DataFrame(columns=["day", "patient_id", "organ_id"])

        self._setup()

    def _setup(self) -> None:
        # RESET WAITLIST TO initial_waitlist_size
        patients_on_waitlist_df = self.DATA.sample(self.initial_waitlist_size)
        waitlist_indxs = patients_on_waitlist_df.index

        self.waitlist = np.array(
            [
                Sim_Patient(
                    id=waitlist_indxs[i],
                    time_to_live=patients_on_waitlist_df.iloc[i].ttl,
                )
                for i in range(len(waitlist_indxs))
            ]
        )

        # RESET STATS
        self.stats = Stats()

    def simulate(self, policy: Policy, log: bool = False) -> Tuple[Stats, pd.DataFrame]:
        # while not stop_critierum
        #   self.iterate(policy)
        have_empty = False
        for day in tqdm(range(self.days)):
            self.iterate(policy, day, log=log)
            if (
                not have_empty
                and day > 0
                and len(self.stats.patients_transplanted[f"{day-1}"])
                < len(self.stats.organs_transplanted[f"{day-1}"])
            ):
                self.stats.first_empty_day = day - 1
                have_empty = True

        return self.stats, self.log_df

    def iterate(self, policy: Policy, _day: int, log: bool = False) -> None:
        log_dict = []  # init log_dict -> each day a new log is added

        dead_patients = self._remove_dead_patients(
            policy
        )  # remove dead patients from waitlist (also in policy)
        amount_died = len(dead_patients)

        if log:
            log_dict.extend(
                [
                    {"patient_id": dead_patient, "organ_id": -1, "day": _day}
                    for dead_patient in dead_patients
                ]
            )  # add dead patients with organ None to log_dict

        patients = self._sample_patients(day=_day)  # sample patient(s)
        organs = self._sample_organs(day=_day)  # sample organ(s)
        policy.add_x(patients)  # add patient(s) and organ(s) to
        transplant_patients = policy.get_xs(organs)  # the policy's internal waitlist
        # and assign organs to patients

        if len(organs) > len(transplant_patients):
            organs = organs[: len(transplant_patients)]
        #  CALCULATE TTL FOR transplant_patients BY policy
        organs_cov = self.organs.loc[
            organs
        ].to_numpy()  # sample organ covariates for sampled indices
        patients_cov = (
            self.patients.loc[  # sample patient covariates for transplanted indices
                transplant_patients
            ].to_numpy()
        )

        if log:  # add transplantation to log_dict
            log_dict.extend(
                [
                    {
                        "patient_id": transplant_patients[r],
                        "organ_id": organs[r],
                        "day": _day,
                    }
                    for r in range(len(transplant_patients))
                ]
            )

        ttl = self.inference_1(x=patients_cov, o=organs_cov).numpy().flatten()

        self._remove_patients(
            transplant_patients
        )  # remove transplanted patients from waitlist
        # note that transplant_patients are automatically
        # removed from the policy's internal waitlist

        # UPDATE STATISTICS
        self.stats.deaths += amount_died
        self.stats.population_life_years += ttl.sum()
        self.stats.transplant_count += len(transplant_patients)

        self.stats.patients_died[f"{_day}"] = dead_patients
        self.stats.patients_transplanted[f"{_day}"] = transplant_patients
        self.stats.organs_transplanted[f"{_day}"] = organs

        if log:  # add log_dict to log_df (later to be returned after simulation ends)
            for log_item in log_dict:
                self.log_df = self.log_df.append(log_item, ignore_index=True)

        self._age_patients(days=30)

    def _remove_patients(self, patients: list) -> None:
        self.waitlist = np.delete(
            self.waitlist,
            np.intersect1d(
                np.array([p.id for p in self.waitlist]), patients, return_indices=True
            )[1],
        )

    def _remove_dead_patients(self, policy: Policy) -> list:

        dead_patients_indices = np.where(
            np.array([p.time_to_live for p in self.waitlist]) <= 0
        )[
            0
        ]  # selects patient IDs when Sim_Patient.ttl <= 0
        tmp = self.waitlist[dead_patients_indices]
        dead_patients_ids = np.array([p.id for p in tmp])

        self._remove_patients(dead_patients_ids)  # remove patients from self.waitlist
        policy.remove_x(
            dead_patients_ids
        )  # remove patients from policy: policy.remove_x(list)

        return dead_patients_ids

    def _age_patients(self, days: int = 1) -> None:
        self.waitlist = np.array([p.age(days) for p in self.waitlist])

    def _sample_patients(self, day: date = None) -> np.ndarray:
        # returns a list of patient IDs
        n = np.abs(
            np.round(  # We sample an amount (n) from a normal distribution
                np.random.normal(
                    loc=self.patient_count / self.days, size=1
                )  # this ensures the average of integers equals that of
            )
        ).item()  # patient_count / days. Note, the scale may be adjusted
        n = int(n)  # should this be too extreme

        patients = self.patients.sample(n=n).index

        new_patients = np.array(
            [
                Sim_Patient(id=patients[i], time_to_live=self.DATA.iloc[i].ttl)
                for i in range(len(patients))
            ]
        )

        self.waitlist = np.append(self.waitlist, new_patients)
        self.waitlist = np.unique(self.waitlist)

        return np.array(patients)

    def _sample_organs(self, day: date = None) -> np.ndarray:
        # returns a list of organ IDs
        n = np.abs(
            np.round(
                np.random.normal(
                    loc=(self.patient_count / self.days) * self.organ_deficit, size=1
                )
            )
        ).item()
        n = int(n)

        organs = self.organs.sample(n=n).index
        return np.array(organs)


class DatedSim(Sim):
    def __init__(
        self,
        dm: OrganDataModule,
        initial_waitlist_size: int,
        inference_0: Inference,
        inference_1: Inference,
        years: int = [2018],
        no_patient_arrival: bool = True,
    ) -> None:

        self.dm = dm

        self.DATA = pd.concat(
            [dm._test_processed.copy(deep=True), dm._train_processed.copy(deep=True)]
        )
        self.DATA_true = self.DATA.copy(deep=True)
        self.DATA_true[self.dm.real_cols] = self.dm.scaler.inverse_transform(
            self.DATA_true[dm.real_cols]
        )
        self.DATA_true["retrieval_date_time"] = pd.to_datetime(
            self.DATA_true.retrieval_date, unit="s"
        )

        self.patients = self.DATA[dm.x_cols].copy(deep=True)
        self.organs = self.DATA[
            np.delete(dm.o_cols, np.where(dm.o_cols == "retrieval_date"))
        ].copy(deep=True)
        self.patients_true = self.DATA_true[dm.x_cols].copy(deep=True)
        self.organs_true = self.DATA_true[
            [
                *np.delete(dm.o_cols, np.where(dm.o_cols == "retrieval_date")),
                "retrieval_date_time",
            ]
        ].copy(deep=True)

        self.years = years
        self.initial_waitlist_size = initial_waitlist_size
        self.start_date = date(np.min(self.years), 1, 1)
        self.end_date = date(np.max(self.years), 12, 31)

        if no_patient_arrival:
            # Ideally, we would also use a day for each patient.
            #   However, this data is absent from the UKReg data
            #   and we have to resort to heuristics. We will assume
            #   a fixed arrival rate of incoming patients and iterate
            #   (in order of appearance) accordingly. This way, the same
            #   order across runs is maintained, and we violate– at most –
            #   a few days of discrepancies with the (unknown) real data.
            patient_indices = np.array(
                self.patients[self.patients_true.regyr.isin(self.years)].index
            )

            amount_of_days = (self.end_date - self.start_date).days + 1
            split_indices = np.array_split(patient_indices, amount_of_days)
            self.DATA_true["patient_arrival_date"] = np.nan

            for i, day in enumerate(
                rrule(DAILY, dtstart=self.start_date, until=self.end_date)
            ):
                self.DATA_true.loc[split_indices[i], "patient_arrival_date"] = day
                self.patients_true.loc[split_indices[i], "patient_arrival_date"] = day

        self.inference_0 = inference_0
        self.inference_1 = inference_1

        self.waitlist = np.array([])

        X_tmp = torch.Tensor(
            self.DATA[self.dm.x_cols].to_numpy()
        )  # Add time to live (ttl) column
        self.DATA.loc[:, "ttl"] = self.inference_0(X_tmp).numpy().flatten()

        self.log_df = pd.DataFrame(columns=["day", "patient_id", "organ_id"])

        self._setup()

    def _setup(self):
        # SETUP SAMPLE
        min_year = np.min(self.years)
        max_year = np.max(self.years)

        indices = self.DATA_true[
            (self.DATA_true.regyr >= min_year - 1) & (self.DATA_true.regyr <= max_year)
        ].index

        self.DATA_true = self.DATA_true.loc[indices]
        self.DATA = self.DATA.loc[indices]

        # SETUP WAITLIST
        waitlist_indxs = (
            self.DATA_true[self.DATA_true.regyr == min_year - 1]
            .tail(self.initial_waitlist_size)
            .index
        )

        patients_on_waitlist_df = self.DATA.loc[waitlist_indxs]

        self.waitlist = np.array(
            [
                Sim_Patient(
                    id=waitlist_indxs[i],
                    time_to_live=patients_on_waitlist_df.iloc[i].ttl,
                )
                for i in range(len(waitlist_indxs))
            ]
        )

        # RESET STATS
        self.stats = Stats()

    def simulate(self, policy: Policy, log: bool = False) -> Tuple[Stats, pd.DataFrame]:

        for day in tqdm(rrule(DAILY, dtstart=self.start_date, until=self.end_date)):
            self.iterate(policy, day, log=log)

        return self.stats, self.log_df

    def _sample_patients(self, day: date = None) -> np.ndarray:
        assert day is not None, "Should have day in DatedSim when sampling patients"

        patient_indices = self.patients_true[
            (pd.DatetimeIndex(self.patients_true.patient_arrival_date).day == day.day)
            & (
                pd.DatetimeIndex(self.patients_true.patient_arrival_date).month
                == day.month
            )
            & (
                pd.DatetimeIndex(self.patients_true.patient_arrival_date).year
                == day.year
            )
        ].index

        return patient_indices

    def _sample_organs(self, day: date = None) -> np.ndarray:
        assert day is not None, "Should have day in DatedSim when sampling organs"

        organ_indices = self.organs_true[
            (pd.DatetimeIndex(self.organs_true.retrieval_date_time).day == day.day)
            & (
                pd.DatetimeIndex(self.organs_true.retrieval_date_time).month
                == day.month
            )
            & (pd.DatetimeIndex(self.organs_true.retrieval_date_time).year == day.year)
        ].index

        return organ_indices
