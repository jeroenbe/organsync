from src.eval_policies.policy import MELD, MELD_na, OrganITE, OrganSync


def main():
    meld = MELD(name='MELD')
    meld_na = MELD_na(name='MELD-na')
    organite = OrganITE(name='OrganITE')
    organsync = OrganSync(name='OrganSync')


if __name__ == "__main__":
    main()