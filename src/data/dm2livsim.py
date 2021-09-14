# DataModule to LivSim translator
# INFO: given a DataModule (dm), we wish to create the
#   input files, required by LivSim.

import pandas as pd


def save_input_files(location, dm):
    # params:
    # :location: str - is the location where the input files will be stored
    # :dm: DataModule - is either UNOS or UKReg or U2U

    # distancetimes.txt
    # -> not considered

    # Donors_Accept.txt
    # -> not considered

    # Donors.txt
    # Replication (take 1) | DSA ID (take 1) | DSA ID (take 1) | Donor Arrival Time (years) | Donor ABO Blood Type | Organ ID | ...add additional features...
    organs = dm._test_processed[dm.o_cols]
    out = pd.DataFrame()

    # DSA_AvgTimes.txt
    # -> not considered

    # Input_Acceptance_Status1.txt
    # -> not considered

    # Input_Geography.txt
    # -> not considered

    # Input_Relist.txt
    # -> not considered

    # Input_SPartners.txt
    # -> not considered

    # Patients.txt
    # Replication (take 1) | Patient ID | DSA ID (take 1) | DSA ID (take 1) | Patient Arrival Time (years) | Patient ABO | Patient Lab MELD | Patient HCC (0, 1) | Status1 (take 0) | Sodium Score | Inactive (0) | ...add additional features...

    # Patients_Accept.txt
    # -> not considered

    # Status.txt
    # -> left stationairy -> not considered

    # status_times.txt
    # -> change with model

    # stepsurvival.txt
    # Step probability | Days survived post-tx | Group probability
    # TODO: check how Step prob. calculates days survived. Can we just fill in days survivied?

    # survivalcoefficients.txt
    # -> change with model

    # Waitlist_matchmeld.txt

    pass
