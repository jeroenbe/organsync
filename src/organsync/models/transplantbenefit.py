from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import organsync.models.third_party.tbs.tbs as tbs_impl

module_path = Path(__file__).parent
path = module_path / Path("third_party/tbs/")


def _parse_reg_year(year: int) -> int:
    if year <= 2007:
        return 0
    elif year == 2008:
        return 1
    elif year == 2009:
        return 2
    elif year == 2010:
        return 3
    elif year == 2011:
        return 4
    else:
        return 5


def _parse_cod(cod: int) -> int:
    if cod == 10 or cod == 11:
        return 0
    elif cod >= 20 and cod <= 29:
        return 1
    return 2


def _parse_dtype(cod: int) -> int:
    if cod == 40 or cod == 41 or cod == 49:
        return 0
    return 1


def _parse_disease_group(disease: int) -> int:
    if disease in [441, 442, 443, 444, 445, 447]:
        return 1
    elif disease in [424]:
        return 2
    elif disease in [419]:
        return 3
    elif disease in [413, 436]:
        return 4
    elif disease in [414]:
        return 5
    elif disease in [411]:
        return 6
    elif disease in [412, 417]:
        return 7
    elif disease in [415, 422, 426, 450, 452, 454, 456, 457, 461, 462, 434]:
        return 8
    elif disease in [
        410,
        416,
        418,
        420,
        421,
        423,
        425,
        448,
        451,
        453,
        455,
        460,
        463,
        464,
        466,
        467,
        468,
        469,
        483,
        484,
        485,
        486,
        498,
        474,
    ]:
        return 9
    return 10


class TBS:
    def fit(self, *args: Any, **kwargs: Any) -> "TBS":
        return self

    def _single_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        centre_tbs = 0
        rregistration_tbs = _parse_reg_year(data["regyr"])
        rinpatient_tbs = int(data["PATIENT_LOCATION"]) - 1
        rwaiting_time_tbs = data["rwtime"] + 1 if "rwtime" in data else 1
        rage_tbs = data["RAGE"]
        rgender_tbs = int(data["SEX"] == 2)
        rdisease_primary_tbs = _parse_disease_group(data["PRIMARY_LIVER_DISEASE"])
        rdisease_secondary_tbs = 9
        rdisease_tertiary_tbs = 9
        previous_tx_tbs = (
            data["NO_OF_PREVIOUS_LIVER_TX"] if "NO_OF_PREVIOUS_LIVER_TX" in data else 0
        )
        rprevious_surgery_tbs = (
            int(data["PREV_ABDOMINAL_SURGERY"])
            if "PREV_ABDOMINAL_SURGERY" in data
            else 8
        )
        rbilirubin_tbs = data["SERUM_BILIRUBIN"]
        rinr_tbs = data["INR"]
        rcreatinine_tbs = data["SERUM_CREATININE"]
        rrenal_tbs = int(data["RENAL_SUPPORT"] != 3)
        rsodium_tbs = data["SERUM_SODIUM"]
        rpotassium_tbs = data["SERUM_POTASSIUM"] if "SERUM_POTASSIUM" in data else 0
        ralbumin_tbs = data["SERUM_ALBUMIN"] if "SERUM_ALBUMIN" in data else 0
        rencephalopathy_tbs = (
            int(data["ENCEPHALOPATHY_GRADE"] != 0)
            if "ENCEPHALOPATHY_GRADE" in data
            else 8
        )
        rascites_tbs = data["CURRENT_ASCITES"] if "CURRENT_ASCITES" in data else 8
        rdiabetes_tbs = int(data["DIABETIC"])
        dage_tbs = data["DAGE"]
        dbmi_tbs = data["DBMI"]
        dcause_tbs = _parse_cod(data["DCOD"])
        ddiabetes_tbs = int(0)
        dtype_tbs = data["DGRP"] == 2
        splittable_tbs = 0
        bloodgroup_compatible_tbs = 1
        rmax_afp_tbs = 5
        rtumour_number_tbs = 0
        rmax_tumour_size_tbs = 1

        output = tbs_impl.fn_tbs(
            centre_tbs,
            rregistration_tbs,
            rinpatient_tbs,
            rwaiting_time_tbs,
            rage_tbs,
            rgender_tbs,
            rdisease_primary_tbs,
            rdisease_secondary_tbs,
            rdisease_tertiary_tbs,
            previous_tx_tbs,
            rprevious_surgery_tbs,
            rbilirubin_tbs,
            rinr_tbs,
            rcreatinine_tbs,
            rrenal_tbs,
            rsodium_tbs,
            rpotassium_tbs,
            ralbumin_tbs,
            rencephalopathy_tbs,
            rascites_tbs,
            rdiabetes_tbs,
            rmax_afp_tbs,
            rtumour_number_tbs,
            rmax_tumour_size_tbs,
            dage_tbs,
            dcause_tbs,
            dbmi_tbs,
            ddiabetes_tbs,
            dtype_tbs,
            splittable_tbs,
            bloodgroup_compatible_tbs,
        )

        return [round(output["tbs"], 1), round(output["m1"], 1), round(output["m2"], 1)]

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        scores = data.apply(lambda row: self._single_predict(row), axis=1)
        scores = np.vstack(scores)
        return pd.DataFrame(scores, columns=["score", "m1", "m2"])
