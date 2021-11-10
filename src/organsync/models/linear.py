import numpy as np


class MELD:
    def score(
        self, serum_bilirubin: float, inr: float, serum_creatinine: float
    ) -> float:
        # DEFINITION OF (standard) MELD: https://en.wikipedia.org/wiki/Model_for_End-Stage_Liver_Disease#Determination
        return (
            3.79 * np.log(serum_bilirubin)  # mg/dL
            + 11.2 * np.log(inr)
            + 9.57 * np.log(serum_creatinine)  # mg / dL
            + 6.43
        )


class MELD_na:
    def score(
        self,
        serum_bilirubin: float,
        inr: float,
        serum_creatinine: float,
        serum_sodium: float,
    ) -> float:
        # MELD-na: MELD + 1.59*(135-SODIUM(mmol/l)) (https://github.com/kartoun/meld-plus/raw/master/MELD_Plus_Calculator.xlsx)
        meld_score = MELD().score(serum_bilirubin, inr, serum_creatinine)

        return (
            meld_score
            - serum_sodium
            - (0.025 * meld_score * (140 - serum_sodium))
            + 140
        )


class UKELD:
    def score(
        self,
        serum_bilirubin: float,
        serum_creatinine: float,
        inr: float,
        serum_sodium: float,
    ) -> float:

        return (
            (5.395 * np.log(inr))
            + (1.485 * np.log(serum_creatinine))
            + (3.13 * np.log(serum_bilirubin))
            - (81.565 * np.log(serum_sodium))
            + 435
        )
