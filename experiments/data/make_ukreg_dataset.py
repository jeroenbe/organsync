from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def load_data(path: Path, lim: Optional[int] = None) -> List[pd.DataFrame]:
    """Loads the data– in parallel –to memory; typically passed to prep_liver_data or any other preprocessor function.
    INPUT - [Iterable[str]]: An Iterable of filepaths to datasets
        * Registration data
        * Sequential data
        * Transplant data
    OUTPUT - Iterable[pd.DataFrame]: Outputs pd.DataFrame for every dataset.
    """
    location = (
        path / "Liver Registrations - Registrations.xlsx",
        path / "Liver Registrations - Sequential.xlsx",
        path / "Liver Registrations - Transplant.xlsx",
    )

    def load(location: Path) -> pd.DataFrame:
        return pd.read_excel(location, nrows=lim)

    return Parallel(n_jobs=len(location))(delayed(load)(loc) for loc in location)


def prep_ukeld_data(
    p: List[pd.DataFrame], replace_organ: int = -1
) -> Tuple[
    pd.DataFrame,
    Iterable[str],
    Iterable[str],
    Iterable[str],
    preprocessing.StandardScaler,
    Iterable[str],
]:

    M1_cols = {
        "PRIMARY_LIVER_DISEASE": False,
        "reg_age": False,
        "SEX": True,
        "SERUM_CREATININE": False,
        "SERUM_BILIRUBIN": False,
        "INR": False,
        "SERUM_SODIUM": False,
        "RENAL_SUPPORT": True,
        "PATIENT_LOCATION": True,
        "DIABETIC": False,
        "regyr": False,
        "outcome": True,
        "rwtime": False,
    }

    X = p[0].loc[:, [*list(M1_cols.keys()), "a_recip_id"]]
    X = X.rename(columns={"a_recip_id": "RECIPID"})

    X.loc[X["outcome"].isin(["A"]), "rwtime"] = np.nan

    for k in M1_cols.keys():
        if M1_cols[k]:
            X.loc[:, k] = X[k].replace(9, np.nan)
            X.loc[:, k] = X[k].replace(8, np.nan)
            X.loc[:, k] = X[k].astype("category")
        else:
            X.loc[:, k] = X[k].replace(888, np.nan)

    X = X.replace([-np.inf, np.inf], np.nan)

    X["CENS"] = X.outcome != "T"
    X["CENS"] = X.CENS.astype(int)
    X = X.drop(columns=["outcome"])
    X["DIABETIC"] = X["DIABETIC"].replace(8, 1)

    X = pd.get_dummies(X)

    x_drop = list(X.loc[:, X.var() < 0.01].columns)

    X = X.drop(columns=x_drop)  # too low variance in col

    M2_cols_X = {
        "RCSPLD1": False,  # 499, 888
        "RAGE": False,
        "RHCV": False,  # 8, 9
        "RCREAT": False,  # 8888, 9999
        "RBILIRUBIN": False,  # 8888, 9999
        "RINR": False,  # 88.8, 99.9
        "RSODIUM": False,  # 888, 999
        "RPOTASSIUM": False,  # 88.8, 99.9
        "RALBUMIN": False,  # 88, 99
        "RREN_SUP": True,  # 8, 9
        "RAB_SURGERY": False,  # 8, 9
        "RENCEPH": True,  # 8, 9
        "RASCITES": True,  # 8, 9
        "PSURV": False,
    }

    M2_cols_D = {
        "DAGE": False,
        "DCOD": True,  # 599, 888
        "DBMI": False,
        "DGRP": False,
        "retrieval_date": False,
    }

    M2_cols = {**M2_cols_X, **M2_cols_D}
    O = p[2].loc[:, [*list(M2_cols.keys()), "RECIPID", "DDATE", "PCENS"]]  # +

    O = O[O.PCENS.notna()]

    for k in M2_cols.keys():
        if M2_cols[k]:
            O.loc[:, k] = O[k].replace(9, np.nan)
            O.loc[:, k] = O[k].replace(8, np.nan)
            O.loc[:, k] = O[k].astype("category")

    O.loc[:, "RCREAT"] = O["RCREAT"].replace([8888, 9999], np.nan)
    O.loc[:, "RBILIRUBIN"] = O["RBILIRUBIN"].replace([8888, 9999], np.nan)
    O.loc[:, "RINR"] = O["RINR"].replace([88.8, 99.9], np.nan)
    O.loc[:, "RSODIUM"] = O["RSODIUM"].replace([888, 999], np.nan)
    O.loc[:, "RPOTASSIUM"] = O["RPOTASSIUM"].replace([88.8, 99.9], np.nan)
    O.loc[:, "RALBUMIN"] = O["RALBUMIN"].replace([88, 99], np.nan)

    O.loc[:, "retrieval_date"] = pd.to_datetime(
        O["retrieval_date"], format="%d%b%y:%H:%M:%S"
    )
    dummy_date = datetime(year=1970, month=1, day=1)
    O.loc[:, "retrieval_date"] = O["retrieval_date"].fillna(dummy_date)
    O.loc[:, "retrieval_date"] = (O["retrieval_date"] - dummy_date).dt.total_seconds()

    O = O.replace([-np.inf, np.inf], np.nan)

    X_drop: List[str] = []

    O_drop = [
        "DCOD_76",
        "DCOD_78",
        "DCOD_80",
    ]

    o_cols = pd.get_dummies(O[list(M2_cols_D.keys())]).drop(O_drop, 1).columns.values
    x_cols = pd.get_dummies(O[list(M2_cols_X.keys())]).drop(X_drop, 1).columns.values

    O = pd.get_dummies(O)
    O = O.drop(columns=[*X_drop, *O_drop])  # drop for too low variance

    DATA = X.merge(O, on="RECIPID", how="left")

    DATA.loc[:, "PSURV"] = DATA["PSURV"].replace(np.nan, 0)
    DATA.loc[:, "rwtime"] = DATA["rwtime"].replace(np.nan, 0)
    DATA.loc[:, "PCENS"] = DATA["PCENS"].replace(np.nan, 0)

    DATA["Y"] = DATA["rwtime"] + DATA["PSURV"]

    impute_cols = list(filter(lambda x: not M1_cols[x], M1_cols)) + list(
        filter(lambda x: not M2_cols[x], M2_cols)
    )
    impute_cols = [*impute_cols, "Y"]

    imputer = IterativeImputer(random_state=0, max_iter=50)  # MICE
    imputer.fit(DATA[impute_cols])
    imputed = imputer.transform(DATA[impute_cols])

    scaler = preprocessing.StandardScaler().fit(imputed)
    scaled = scaler.transform(imputed)

    DATA.loc[:, impute_cols] = scaled
    DATA = DATA.drop(columns=["RECIPID"])
    X = X.drop(columns=["RECIPID"])

    return DATA, X.columns.values, x_cols, o_cols, scaler, impute_cols[:-1]


def save_data(dfs: dict, models: dict, nps: dict, loc: Path) -> None:
    for k, v in dfs.items():
        v.to_csv(f"{loc}/{k}.csv")

    for k, v in models.items():
        dump(v, f"{loc}/{k}")

    for k, v in nps.items():
        np.save(f"{loc}/{k}.npy", v)


@click.command()
@click.option("-o", "--output_dir", type=str)
@click.option("-i", "--input_dir", type=str)
def cli(output_dir: str, input_dir: str) -> None:
    output_path = Path(output_dir)
    input_path = Path(input_dir)

    data = load_data(path=input_path)

    d, xm1, xm2, om2, scaler, i = prep_ukeld_data(p=data)

    dfs = {"data_preprocessed": d}
    models = {"scaler": scaler}
    nps = {"impute": i, "x_cols_m1": xm1, "x_cols_m2": xm2, "o_cols_m2": om2}

    if not output_path.exists():
        output_path.mkdir()
    save_data(dfs=dfs, models=models, nps=nps, loc=output_path)


if __name__ == "__main__":
    cli()