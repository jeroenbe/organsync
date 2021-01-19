#######################################################################################Entity Classes#############################################################################
class Patient:
    """
    This is the Patient class. It records information regarding the Patient
    """
    ABO=-1 #blood type
    MELD=-1 #allocation MELD score
    lMELD =-1 #lab MELD score
    HCC=-1 #HCC exception status of patient
    Status1 =-1 #status 1 indicator
    Na = -1 #sodium score of patient
    Inactive = 0 #waitlist inactive status of patient
    Relist = 0 #indicator of whether patient is relisted or not
    RelistTxTime  = 0 #time at which patient is eligible for re-transplant


    #Patient Attributes for Constructor
    def __init__(self,patient_id,patient_DSA,patient_createtime):
        self.id = patient_id #patient id
        self.DSA = patient_DSA #DSA that the patient belongs to
        self.create_time = patient_createtime #time when patient was created

class Organ:
    """
    This is the organ class. It records information about the donated organ.
    """
    #Organ Attributes 
    ABO =-1 #blood type
    organid =-1 #organ id
    #Organ Attributes for Constructor
    def __init__(self,organ_DSA):
        self.DSA = organ_DSA #DSA that organ belongs to

###############################################################################################################################################################################################