import numpy as np


class MELD:
    def score(
        self, serum_bilirubin: np.ndarray, inr: np.ndarray, serum_creatinine: np.ndarray
    ) -> np.ndarray:
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
        serum_bilirubin: np.ndarray,
        inr: np.ndarray,
        serum_creatinine: np.ndarray,
        serum_sodium: np.ndarray,
    ) -> np.ndarray:
        # MELD-na: MELD + 1.59*(135-SODIUM(mmol/l)) (https://github.com/kartoun/meld-plus/raw/master/MELD_Plus_Calculator.xlsx)
        return MELD().score(serum_bilirubin, inr, serum_creatinine) + 1.59 * (
            135 - serum_sodium  # mmol / L
        )
