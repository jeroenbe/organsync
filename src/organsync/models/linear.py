import numpy as np


class MELD:
    def score(
        self,
        serum_bilirubin: float,
        inr: float,
        serum_creatinine: float,
    ) -> float:
        # DEFINITION OF (standard) MELD: https://en.wikipedia.org/wiki/Model_for_End-Stage_Liver_Disease#Determination
        serum_bilirubin = max(serum_bilirubin, 1)
        inr = max(inr, 1)
        serum_creatinine = max(serum_creatinine, 1)
        serum_creatinine = min(serum_creatinine, 4)

        score = (
            3.78 * np.log(serum_bilirubin)  # mg/dL
            + 11.2 * np.log(inr)
            + 9.57 * np.log(serum_creatinine)  # mg / dL
            + 6.43
        )
        if score < 6:
            return 6

        return score


class MELD_na:
    def score(
        self,
        serum_bilirubin: float,  # mg/dL
        inr: float,
        serum_creatinine: float,  # mg / dL
        serum_sodium: float,  # mmol/L
    ) -> float:
        # "Hyponatremia and Mortality among Patients on the LiverTransplant Waiting List"
        # MELD-na: MELD - Na - (0.0.25 * MELD * (140 - Na)) + 140
        #
        # MELD 3.0: The Model for End-Stage Liver Disease Updated for the Modern Era
        # MELD-na: MELD + [1.32 * (137 - Na)] - [0.033 * MELD * (137 - Na)]

        meld_score = MELD().score(serum_bilirubin, inr, serum_creatinine)

        serum_sodium = max(serum_sodium, 125)
        serum_sodium = min(serum_sodium, 137)

        score = (
            meld_score
            + (1.32 * (137 - serum_sodium))
            - (0.033 * meld_score * (137 - serum_sodium))
        )
        if score < 6:
            return 6

        return score


class MELD3:
    # MELD 3.0: The Model for End-Stage Liver Disease Updated for the Modern Era
    def score(
        self,
        sex: str,  # "M" or "F"
        serum_bilirubin: float,  # mg/dL
        inr: float,
        serum_creatinine: float,  # mg / dL
        serum_sodium: float,  # mmol/L
        serum_albumin: float,  # g/dL
    ) -> float:
        # MELD3 =
        #    1.33 * (is female)
        #   + [4.56 * log (bilirubin)] + [0.82 *  (137 – Na)]
        #   – [0.24  (137 – Na) * log(bilirubin)]
        #   + [9.09 * log(INR)] + [11.14 * log(creatinine)]
        #   + [1.85 * (3.5 – albumin)] – [1.83 * (3.5 – albumin) * log(creatinine)] + 6
        if sex not in ["M", "F"]:
            raise ValueError("Sex must be from ['M', 'F']")

        serum_bilirubin = max(serum_bilirubin, 1)
        inr = max(inr, 1)

        serum_creatinine = max(serum_creatinine, 1)
        serum_creatinine = min(serum_creatinine, 4)

        serum_sodium = max(serum_sodium, 125)
        serum_sodium = min(serum_sodium, 137)

        serum_albumin = max(serum_albumin, 2.7)
        serum_albumin = min(serum_albumin, 3.6)

        score = 1.33 if sex == "F" else 0
        score += (
            (4.56 * np.log(serum_bilirubin))
            + (0.82 * (137 - serum_sodium))
            - (0.24 * (137 - serum_sodium) * np.log(serum_bilirubin))
            + (9.09 * np.log(inr))
            + (11.14 * np.log(serum_creatinine))
            + (1.85 * (3.5 - serum_albumin))
            - (1.83 * (3.5 - serum_albumin) * np.log(serum_creatinine))
            + 6
        )

        return score


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
