# ported from tbs.R
import typing
from pathlib import Path

import numpy as np
import pandas as pd

module_path = Path(__file__).parent
betas = pd.read_csv(module_path / "betas.csv")
surv_noncancer = pd.read_csv(module_path / "surv_noncancer.csv")
surv_cancer = pd.read_csv(module_path / "surv_cancer.csv")
pd.options.mode.chained_assignment = None  # default='warn'

EPS = 1e-8


def idx2dummy(group: int, length: int) -> list:
    if group > length:
        print("group, length", group, length, flush=True)
        raise ValueError("Group must be an integer less than or equal to length")

    out = np.zeros(length)
    out[group] = 1

    return out.tolist()


def make_rdisease_vec(
    rdisease_primary_tbs: int,
    rdisease_secondary_tbs: int,
    rdisease_tertiary_tbs: int,
    previous_tx_tbs: int,
) -> list:
    if previous_tx_tbs > 0:
        out = idx2dummy(9, 10)
    else:
        for code in range(1, 11):
            if (
                rdisease_primary_tbs == code
                or rdisease_secondary_tbs == code
                or rdisease_tertiary_tbs == code
            ):
                out = idx2dummy(code - 1, 10)
                break

    del out[2]
    del out[0]

    return out


def is_rhcv(
    rdisease_primary_tbs: int, rdisease_secondary_tbs: int, rdisease_tertiary_tbs: int
) -> bool:
    return (
        rdisease_primary_tbs == 2
        or rdisease_secondary_tbs == 2
        or rdisease_tertiary_tbs == 2
    )


def is_cancer(
    rdisease_primary_tbs: int, rdisease_secondary_tbs: int, rdisease_tertiary_tbs: int
) -> bool:
    return (
        rdisease_primary_tbs == 1
        or rdisease_secondary_tbs == 1
        or rdisease_tertiary_tbs == 1
    )


def make_rcreatinine(centre_tbs: int, rcreatinine_tbs: float) -> float:
    if centre_tbs == 3:
        return (rcreatinine_tbs + 23.4) / 1.2

    return rcreatinine_tbs


@typing.no_type_check
def make_x1(
    rage_tbs: int,
    rgender_tbs: int,
    rhcv: int,
    rdisease_vec: list,
    rcreatinine: float,
    rbilirubin_tbs: float,
    rinr_tbs: float,
    rsodium_tbs: float,
    rpotassium_tbs: float,
    ralbumin_tbs: float,
    rrenal_tbs: int,
    rinpatient_tbs: int,
    rregistration_vec: list,
    rprevious_surgery_tbs: int,
    rencephalopathy_tbs: int,
    rascites_tbs: int,
    rwaiting_time_tbs: int,
    rdiabetes_tbs: int,
    rmax_afp_tbs: int,
    rmax_tumour_size_tbs: int,
    rtumour_number_vec: list,
    dage_tbs: int,
    dcause_vec: list,
    dbmi_tbs: float,
    ddiabetes_vec: list,
    dtype_tbs: int,
    bloodgroup_compatible_tbs: int,
    splittable_tbs: int,
) -> np.ndarray:
    return np.asarray(
        [rage_tbs, rage_tbs, rgender_tbs, rhcv]
        + rdisease_vec
        + [
            rcreatinine,
            rbilirubin_tbs,
            rinr_tbs,
            rsodium_tbs,
            rpotassium_tbs,
            ralbumin_tbs,
            rrenal_tbs,
            rinpatient_tbs,
        ]
        + rregistration_vec
        + [rbilirubin_tbs]
        + rdisease_vec
        + [
            rage_tbs,
            rprevious_surgery_tbs,
            rencephalopathy_tbs,
            rascites_tbs,
            int(rwaiting_time_tbs) + 1,
            rdiabetes_tbs,
        ]
        + rdisease_vec
        + [int(rmax_afp_tbs) + 1, rmax_tumour_size_tbs]
        + rtumour_number_vec
        + [dage_tbs]
        + dcause_vec
        + [dbmi_tbs]
        + ddiabetes_vec
        + [dtype_tbs, rhcv, rhcv, rhcv, dtype_tbs, dtype_tbs]
        + rdisease_vec
        + [
            bloodgroup_compatible_tbs,
            splittable_tbs,
        ]
    )


def make_x2(
    rsodium_tbs: float,
    rbilirubin_tbs: float,
    rcreatinine: float,
    rage_tbs: int,
    ddiabetes_vec: list,
    dage_tbs: int,
    dtype_tbs: int,
) -> np.ndarray:
    return np.asarray(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            rsodium_tbs,
            rbilirubin_tbs,
            rbilirubin_tbs,
            rbilirubin_tbs,
            rbilirubin_tbs,
            rbilirubin_tbs,
            rbilirubin_tbs,
            rbilirubin_tbs,
            rbilirubin_tbs,
            rcreatinine,
            1,
            1,
            1,
            1,
            1,
            rage_tbs,
            rage_tbs,
            rage_tbs,
            rage_tbs,
            rage_tbs,
            rage_tbs,
            rage_tbs,
            rage_tbs,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
        + ddiabetes_vec
        + [
            dage_tbs,
            rage_tbs,
            rcreatinine,
            dtype_tbs,
            dtype_tbs,
            dtype_tbs,
            dtype_tbs,
            dtype_tbs,
            dtype_tbs,
            dtype_tbs,
            dtype_tbs,
            1,
            1,
        ]
    )


def fn_tbs(
    centre_tbs: int,
    rregistration_tbs: int,
    rinpatient_tbs: int,
    rwaiting_time_tbs: int,
    rage_tbs: int,
    rgender_tbs: int,
    rdisease_primary_tbs: int,
    rdisease_secondary_tbs: int,
    rdisease_tertiary_tbs: int,
    previous_tx_tbs: int,
    rprevious_surgery_tbs: int,
    rbilirubin_tbs: float,
    rinr_tbs: float,
    rcreatinine_tbs: float,
    rrenal_tbs: int,
    rsodium_tbs: float,
    rpotassium_tbs: float,
    ralbumin_tbs: float,
    rencephalopathy_tbs: int,
    rascites_tbs: int,
    rdiabetes_tbs: int,
    rmax_afp_tbs: int,
    rtumour_number_tbs: int,
    rmax_tumour_size_tbs: int,
    dage_tbs: int,
    dcause_tbs: int,
    dbmi_tbs: float,
    ddiabetes_tbs: int,
    dtype_tbs: int,
    splittable_tbs: int,
    bloodgroup_compatible_tbs: int,
) -> dict:
    rregistration_vec = idx2dummy(int(rregistration_tbs), 6)
    rdisease_vec = make_rdisease_vec(
        rdisease_primary_tbs,
        rdisease_secondary_tbs,
        rdisease_tertiary_tbs,
        previous_tx_tbs,
    )
    rcancer = is_cancer(
        rdisease_primary_tbs, rdisease_secondary_tbs, rdisease_tertiary_tbs
    )
    rhcv = is_rhcv(rdisease_primary_tbs, rdisease_secondary_tbs, rdisease_tertiary_tbs)
    rcreatinine = make_rcreatinine(centre_tbs, rcreatinine_tbs)
    rtumour_number_vec = idx2dummy(int(rtumour_number_tbs), 2)
    dcause_vec = idx2dummy(int(dcause_tbs), 3)
    ddiabetes_vec = idx2dummy(int(ddiabetes_tbs), 2)

    x1 = make_x1(
        rage_tbs,
        rgender_tbs,
        rhcv,
        rdisease_vec,
        rcreatinine,
        rbilirubin_tbs,
        rinr_tbs,
        rsodium_tbs,
        rpotassium_tbs,
        ralbumin_tbs,
        rrenal_tbs,
        rinpatient_tbs,
        rregistration_vec,
        rprevious_surgery_tbs,
        rencephalopathy_tbs,
        rascites_tbs,
        rwaiting_time_tbs,
        rdiabetes_tbs,
        rmax_afp_tbs,
        rmax_tumour_size_tbs,
        rtumour_number_vec,
        dage_tbs,
        dcause_vec,
        dbmi_tbs,
        ddiabetes_vec,
        dtype_tbs,
        bloodgroup_compatible_tbs,
        splittable_tbs,
    )
    x2 = make_x2(
        rsodium_tbs,
        rbilirubin_tbs,
        rcreatinine,
        rage_tbs,
        ddiabetes_vec,
        dage_tbs,
        dtype_tbs,
    )

    # Make big prediction table
    linear_prediction = betas.copy()
    linear_prediction["raw_x1"] = x1
    linear_prediction["raw_x2"] = x2

    linear_prediction["transformed_x1"] = linear_prediction["raw_x1"].pow(
        linear_prediction["power"]
    )
    linear_prediction["transformed_x1"][linear_prediction["ln_1"] == 1] = np.log(
        linear_prediction["transformed_x1"] + EPS
    )

    linear_prediction["transformed_x2"] = linear_prediction["raw_x2"].pow(
        linear_prediction["power"]
    )
    linear_prediction["transformed_x2"][linear_prediction["ln_2"] == 1] = np.log(
        linear_prediction["transformed_x2"] + EPS
    )

    if rcancer:
        working_df = linear_prediction[
            [
                "parameter",
                "m1_cancer_beta",
                "m2_cancer_beta",
                "m1_cancer_mean",
                "m2_cancer_mean",
                "raw_x1",
                "raw_x2",
                "transformed_x1",
                "transformed_x2",
            ]
        ].rename(
            columns={
                "m1_cancer_beta": "m1_beta",
                "m2_cancer_beta": "m2_beta",
                "m1_cancer_mean": "m1_mean",
                "m2_cancer_mean": "m2_mean",
            }
        )
    else:
        working_df = linear_prediction[
            [
                "parameter",
                "m1_noncancer_beta",
                "m2_noncancer_beta",
                "m1_noncancer_mean",
                "m2_noncancer_mean",
                "raw_x1",
                "raw_x2",
                "transformed_x1",
                "transformed_x2",
            ]
        ].rename(
            columns={
                "m1_noncancer_beta": "m1_beta",
                "m2_noncancer_beta": "m2_beta",
                "m1_noncancer_mean": "m1_mean",
                "m2_noncancer_mean": "m2_mean",
            }
        )

    working_df["m1_x"] = (
        working_df["transformed_x1"] * working_df["transformed_x2"]
        - working_df["m1_mean"]
    )
    working_df["m2_x"] = (
        working_df["transformed_x1"] * working_df["transformed_x2"]
        - working_df["m2_mean"]
    )
    working_df["m1_beta_x"] = working_df["m1_beta"] * working_df["m1_x"]
    working_df["m2_beta_x"] = working_df["m2_beta"] * working_df["m2_x"]

    # Sum predictor
    m1_linear_predictor = working_df["m1_beta_x"].sum()
    m2_linear_predictor = working_df["m2_beta_x"].sum()

    # Pick appropriate survival table
    surv_active = surv_cancer if rcancer else surv_noncancer

    # Generate need, utility and TBS
    m1 = surv_active["m1_surv"].pow(np.exp(m1_linear_predictor)).sum()
    m2 = surv_active["m2_surv"].pow(np.exp(m2_linear_predictor)).sum()
    tbs = m2 - m1

    return {
        "m1": m1,
        "m2": m2,
        "tbs": tbs,
    }
