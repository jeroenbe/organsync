import numpy as np
import pandas as pd

x_cols = [
    "AGE",
    "ALBUMIN_TX",
    "ASCITES_TX",
    "BACT_PERIT_TCR",
    "BMI_DON_CALC",
    "CITIZENSHIP",
    "CMV_STATUS",
    "CREAT_TX",
    "DAYSWAIT_CHRON",
    "DGN2_TCR2",
    "DGN_TCR1",
    "DIAL_TX",
    "EBV_SEROSTATUS",
    "EDUCATION",
    "ENCEPH_TX",
    "ETHCAT",
    "ETHNICITY",
    "EVER_APPROVED",
    "EXC_CASE",
    "EXC_DIAG_ID",
    "EXC_HCC",
    "FINAL_INR",
    "FINAL_SERUM_SODIUM",
    "FUNC_STAT_TCR",
    "FUNC_STAT_TRR",
    "GENDER",
    "HBV_CORE",
    "HBV_SUR_ANTIGEN",
    "HCC_DIAG",
    "HCC_DIAGNOSIS_TCR",
    "HCC_EVER_APPR",
    "HCV_SEROSTATUS",
    "HGT_CM_CALC",
    "HGT_CM_TCR",
    "HIV_SEROSTATUS",
    "INIT_ALBUMIN",
    "INIT_ASCITES",
    "INIT_BILIRUBIN",
    "INIT_BMI_CALC",
    "INIT_DIALYSIS_PRIOR_WEEK",
    "INIT_ENCEPH",
    "INIT_INR",
    "INIT_MELD_PELD_LAB_SCORE",
    "INIT_SERUM_CREAT",
    "INIT_SERUM_SODIUM",
    "INIT_WGT_KG",
    "INR_TX",
    "LIFE_SUP_TCR",
    "LIFE_SUP_TRR",
    "MALIG_TCR",
    "MALIG_TRR",
    "MED_COND_TRR",
    "MELD_PELD_LAB_SCORE",
    "NUM_PREV_TX",
    "ON_VENT_TRR",
    "PORTAL_VEIN_TCR",
    "PORTAL_VEIN_TRR",
    "PREV_AB_SURG_TCR",
    "PREV_AB_SURG_TRR",
    "PREV_TX",
    "TBILI_TX",
    "TIPSS_TCR",
    "TIPSS_TRR",
    "VENTILATOR_TCR",
    "WGT_KG_CALC",
    "WORK_INCOME_TCR",
    "WORK_INCOME_TRR",
    "abo",
    "cancer",
    "diabbin",
    "diag1",
    "hosptime",
    "insdiab",
    "meldstat",
    "meldstatinit",
    "status1",
    "statushcc",
]

o_cols = [
    "AGE_DON",
    "ALCOHOL_HEAVY_DON",
    "ANTIHYPE_DON",
    "ARGININE_DON",
    "BLOOD_INF_DON",
    "BMI_DON_CALC",
    "BUN_DON",
    "CARDARREST_NEURO",
    "CDC_RISK_HIV_DON",
    "CITIZENSHIP_DON",
    "CLIN_INFECT_DON",
    "CMV_DON",
    "COD_CAD_DON",
    "CREAT_DON",
    "DDAVP_DON",
    "DIABETES_DON",
    "EBV_IGG_CAD_DON",
    "EBV_IGM_CAD_DON",
    "ETHCAT_DON",
    "GENDER_DON",
    "HBSAB_DON",
    "HBV_CORE_DON",
    "HEMATOCRIT_DON",
    "HEPARIN_DON",
    "HEP_C_ANTI_DON",
    "HGT_CM_DON_CALC",
    "HISTORY_MI_DON",
    "HIST_CANCER_DON",
    "HIST_CIG_DON",
    "HIST_COCAINE_DON",
    "HIST_HYPERTENS_DON",
    "HIST_INSULIN_DEP_DON",
    "HIST_OTH_DRUG_DON",
    "INOTROP_SUPPORT_DON",
    "INSULIN_DON",
    "INTRACRANIAL_CANCER_DON",
    "NON_HRT_DON",
    "PH_DON",
    "PREV_TX_ANY",
    "PROTEIN_URINE",
    "PT_DIURETICS_DON",
    "PT_STEROIDS_DON",
    "PT_T3_DON",
    "PT_T4_DON",
    "PULM_INF_DON",
    "RECOV_OUT_US",
    "RESUSCIT_DUR",
    "SGOT_DON",
    "SGPT_DON",
    "SKIN_CANCER_DON",
    "TATTOOS",
    "TBILI_DON",
    "TRANSFUS_TERM_DON",
    "URINE_INF_DON",
    "VASODIL_DON",
    "VDRL_DON",
    "WGT_KG_DON_CALC",
    "abodon",
    "coronary",
    "death_mech_don_group",
    "deathcirc",
    "dontime",
    "macro",
    "micro",
]


x_cols_unos_ukeld = [
    "diag1",  #'PRIMARY_LIVER_DISEASE' TODO: see corresponding with UKReg (same codes)
    "AGE",  #'reg_age'
    "GENDER",  #'SEX'
    "INIT_SERUM_CREAT",  #'SERUM_CREATININE'
    "INIT_BILIRUBIN",  #'SERUM_BILIRUBIN'
    "INIT_INR",  #'INR'
    "INIT_SERUM_SODIUM",  #'SERUM_SODIUM'
    "INIT_DIALYSIS_PRIOR_WEEK",
    "DIAL_TX",  #'RENAL_SUPPORT', 'RREN_SUP'
    "MED_COND_TRR",  #'PATIENT_LOCATION' TODO: check whether variables correspond to UKReg
    "LISTYR",  #'regyr'
    # , #'outcome'
    # , #'RCSPLD1' -> diag1 (above)
    # , #'RAGE'
    # , #'RSEX'
    "HCV_SEROSTATUS",  #'RHCV'
    "CREAT_TX",  #'RCREAT'
    "TBILI_TX",  #'RBILIRUBIN'
    "FINAL_INR",  #'RINR'
    "FINAL_SERUM_SODIUM",  #'RSODIUM'
    # , #'RPOTASSIUM' -> not in UNOS
    "INIT_ALBUMIN",  #'RALBUMIN'
    #'PREV_AB_SURG_TCR',
    "PREV_AB_SURG_TRR",  #'RAB_SURGERY'
    "INIT_ENCEPH",  #'RENCEPH'
    #'INIT_ASCITES',
    "ASCITES_TX",  #'RASCITES' TODO: this is categorical in UNOS, see how this corresponds to UKReg
    # , #'PSURV'
]

o_cols_unos_ukeld = [
    "AGE_DON",  #'DAGE'
    #'death_mech_don_group', 'deathcirc', #'DCOD' -> TODO: these may vary significantly from UKReg, best to check
    "BMI_DON_CALC",  #'DBMI'
    "NON_HRT_DON",  #'DGRP' -> in UNOS 'NON_HRT_DON' distinguishes between dead donors and circulatory death donors;
    # Living are excluded for now. TODO: check wether codes correspond to UKReg
]

UNOS_2_UKReg_mapping = {
    "AGE": "RAGE",
    "AGE_DON": "DAGE",
    "ASCITES_TX": "RASCITES",  # TODO: merge
    "BMI_DON_CALC": "DBMI",
    "CREAT_TX": "RCREAT",
    "DIAL_TX": "RREN_SUP",  # , RREN_SUP TODO: merge
    "FINAL_INR": "RINR",
    "FINAL_SERUM_SODIUM": "RSODIUM",
    "GENDER": "SEX",
    "HCV_SEROSTATUS": "RHCV",
    "INIT_ALBUMIN": "RALBUMIN",
    #'INIT_ASCITES':             'RASCITES',                 # TODO: merge
    "INIT_BILIRUBIN": "SERUM_BILIRUBIN",
    "INIT_DIALYSIS_PRIOR_WEEK": "RENAL_SUPPORT",  # , RREN_SUP TODO: merge
    "INIT_ENCEPH": "RENCEPH",
    "INIT_INR": "INR",
    "INIT_SERUM_CREAT": "SERUM_CREATININE",
    "INIT_SERUM_SODIUM": "SERUM_SODIUM",
    "LISTYR": "regyr",
    "MED_COND_TRR": "PATIENT_LOCATION",
    "NON_HRT_DON": "DGRP",
    #'PREV_AB_SURG_TCR':         'RAB_SURGERY',              # TODO: merge
    "PREV_AB_SURG_TRR": "RAB_SURGERY",  # TODO: merge
    "PTIME": "rwtime",
    "TBILI_TX": "RBILIRUBIN",
    #'death_mech_don_group':     'DCOD',                     # TODO: merge
    #'deathcirc':                'DCOD',                     # TODO: merge
    "diag1": "PRIMARY_LIVER_DISEASE",
    "PSTATUS": "CENS",
}


def get_data_tuples(location: str) -> tuple:
    # Divide data in tuples as described in the paper.

    # LOAD
    liver_train = pd.read_csv(f"{location}/liver_processed_train.csv")
    liver_test = pd.read_csv(f"{location}/liver_processed_test.csv")

    # ONLY USE PRESENT COLS
    x_cols_intersected = np.intersect1d(liver_train.columns.values, x_cols)
    o_cols_intersected = np.intersect1d(liver_train.columns.values, o_cols)

    # SPLIT FILE IN PATIENTS (X), ORGANS (O), OUTCOME (Y), CENSOR (del)
    X_train = liver_train[x_cols_intersected]
    O_train = liver_train[o_cols_intersected]
    Y_train = liver_train[["PTIME"]]
    del_train = liver_train[["PSTATUS"]] - 1  # PSTATUS is a censor variable

    X_test = liver_test[x_cols_intersected]
    O_test = liver_test[o_cols_intersected]
    Y_test = liver_test[["PTIME"]]
    del_test = liver_test[["PSTATUS"]] - 1

    return X_train, O_train, Y_train, del_train, X_test, O_test, Y_test, del_test
