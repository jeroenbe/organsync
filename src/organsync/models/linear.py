import numpy as np


class MELD:
    def score(
        self,
        serum_bilirubin: float,
        inr: float,
        serum_creatinine: float,
    ) -> float:
        # DEFINITION OF (standard) MELD: https://en.wikipedia.org/wiki/Model_for_End-Stage_Liver_Disease#Determination
        serum_bilirubin = np.clip(serum_bilirubin, 1, 10)
        inr = np.clip(inr, 1, 10)
        serum_creatinine = np.clip(serum_creatinine, 1, 4)

        score = (
            3.78 * np.log(serum_bilirubin)  # mg/dL
            + 11.2 * np.log(inr)
            + 9.57 * np.log(serum_creatinine)  # mg / dL
            + 6.43
        )
        return score


class MELD_na:
    def score(
        self,
        serum_bilirubin: float,  # mg/dL
        inr: float,
        serum_creatinine: float,  # mg / dL
        serum_sodium: float,  # mmol/L
    ) -> float:
        # Variants od MELDna
        #
        # (https://github.com/kartoun/meld-plus/raw/master/MELD_Plus_Calculator.xlsx)
        # MELD-na: MELD + 1.59*(135-SODIUM(mmol/l))
        #
        # "Hyponatremia and Mortality among Patients on the LiverTransplant Waiting List"
        # MELD-na: MELD - Na - (0.0.25 * MELD * (140 - Na)) + 140
        #
        # MELD 3.0: The Model for End-Stage Liver Disease Updated for the Modern Era
        # MELD-na: MELD + [1.32 * (137 - Na)] - [0.033 * MELD * (137 - Na)]

        meld_score = MELD().score(serum_bilirubin, inr, serum_creatinine)

        serum_sodium = np.clip(serum_sodium, 125, 137)

        score = (
            meld_score
            + (1.32 * (137 - serum_sodium))
            - (0.033 * meld_score * (137 - serum_sodium))
        )
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
        if not set(sex).issubset(set(["M", "F"])):
            raise ValueError(f"Sex must be from ['M', 'F'] : {sex}")

        serum_bilirubin = np.clip(serum_bilirubin, 1, 10)
        inr = np.clip(inr, 1, 10)
        serum_creatinine = np.clip(serum_creatinine, 1, 4)
        serum_sodium = np.clip(serum_sodium, 125, 137)
        serum_albumin = np.clip(serum_albumin, 2.7, 3.6)

        sex_score = np.asarray(sex)
        sex_score[sex_score == "F"] = 1.33
        sex_score[sex_score == "M"] = 0

        score = sex_score.astype(float)
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
        inr = np.clip(inr, 1, 10)
        serum_creatinine = np.clip(serum_creatinine, 1, 4)
        serum_bilirubin = np.clip(serum_bilirubin, 1, 10)
        serum_sodium = np.clip(serum_sodium, 125, 137)

        return (
            (5.395 * np.log(inr))
            + (1.485 * np.log(serum_creatinine))
            + (3.13 * np.log(serum_bilirubin))
            - (81.565 * np.log(serum_sodium))
            + 435
        )
