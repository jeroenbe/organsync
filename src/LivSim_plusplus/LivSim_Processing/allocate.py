from copy import deepcopy
import numpy as nump
ndsa = 58
#####################################################Organ Allocation Procedures, Offer Routines, and Matching Functions#######################################################################
def Allocate(organ, OPTN, Sim, Regions, SharingPartners, Patients_Accept, Donor_Accept, DSA_Avg_Times, AcceptanceModelS1, AcceptanceModel):
    """
    This function compiles the offer lists and calls functions for subsequent matching and offering. It returns a tuple describing whether the organ is transplanted
    or discarded, the transplanting DSA (if transplanted), and the patient id (if transplanted).
    Input:
        @organ: organ that needs a match for transplant
        @OPTN: complete patient data
        @Sim: class object containing relevant variables for simulation
        @Regions: neighborhood map for regions, districts, or neighborhoods
        @SharingPartners: neighbhoord map adding sharing partners to existing geographic relationships among OPOs
        @Patients_Accept: coefficients regarding patient's characteristics for acceptance model
        @Donor_Accept: coefficients regarding donor's characteristics for acceptance model
        @DSA_Avg_Times: data on average transport times between DSAs
        @AcceptanceModelS1: coefficients regarding patient's characteristics for status 1 acceptance model
        @AcceptanceModel: coefficients regarding patient's characteristics for non-status 1 acceptance model
    Output:
        @(organ transplanted/discarded, transplanting DSA [if transplanted], patient id [if transplanted]): tupe with information on the organ and 
        corresponding DSA and patient (if transplanted)
    """
    #Compile Offer List
    LocalList = deepcopy(OPTN[organ.DSA]) #preinitialize list of potential match patients within the DSA; list of match patients already made
    RegionalList = [] #preinitialize list of potential match patients within a region
    NationalList =[] #preinitialize list of potential match patients outside a region within the nation

    #Give boost to local candidates if applicable
    if Sim.localboost > 0:
        #iterate through patients of local list
        for patient in LocalList:

            #if patient MELD is below 40, give a local boost
            if patient.MELD <=40:
                
                #Boosting non-Status1 candidates
                patient.MELD = patient.MELD + Sim.localboost
                
                #if MELD score is over 40, set it down to 40 as the max
                if patient.MELD > 40:
                    patient.MELD =40


    #iterate through list of DSAs
    for i in range(0,ndsa):

        #if sharing partners are implemented and if a DSA is a sharing partner of the current DSA
        #add to the regional list
        if Sim.spartners ==1 and SharingPartners[organ.DSA,i]==1:
            RegionalList = RegionalList + deepcopy(OPTN[i])

        #if a DSA is a neighbor of a current DSA, add it to the regional list
        if Regions[organ.DSA,i] ==1 and i !=organ.DSA:
            RegionalList = RegionalList + deepcopy(OPTN[i])

        #if not, add it to the national list
        elif Regions[organ.DSA,i] !=1 and i !=organ.DSA:
            NationalList = NationalList + deepcopy(OPTN[i])

    #Give boost to regional candidates if applicable
    if Sim.regionalboost > 0:
        for patient in RegionalList:
            if patient.MELD <=40:
                #Boosting non-Status1 candidates
                patient.MELD = patient.MELD + Sim.regionalboost
                if patient.MELD > 40:
                    patient.MELD =40

    #If there is regional sharing, merge the local list and the regional list together
    if Sim.regional_sharing == 1:

        #combine the local list and regional list
        OfferList = LocalList + RegionalList

        #sort the National List
        NationalList = sorted(NationalList,key=lambda patient: patient.MELD, reverse=True)

        #sort the merged list
        OfferList = sorted(OfferList,key=lambda patient: patient.MELD, reverse=True)
        
        #combine the offer list and national list together
        OfferList = OfferList + NationalList


    else:
        #Implement Share 35 and Share 15
        OfferList = [] #preintialize Offer List
        Share35 = [] #preinitialize Share 35 list
        StandardList = [] #preintialize Standard List
        Share15 = [] #preinitialize Share 15 list
        Share15_2 = [] #preinitialize another Share15 List
        Share15_3 = [] #preinitialize another Share15 List
        StandardList2 = [] #preinitialize another standard list
        StandardList3 = [] #preinitialize another standard list
        #iterate through patients of the Local List
        for patient in LocalList:

            #if patient's MELD is at least Share35 value, add him to the Share35 List
            if patient.MELD >= Sim.ShareU:
                Share35.append(patient)

            #if patient's MELD is below Share15 value, add him to the Share 15 List
            elif patient.MELD <Sim.ShareL:
                Share15.append(patient)

            #else add him to the Standard List
            else:
                StandardList.append(patient)

        #iterate through patients of the Regional List
        for patient in RegionalList:

            #if patient's MELD is at least Share35 value, add him to the Share35 List
            if patient.MELD >= Sim.ShareU:
                Share35.append(patient)

            #if patient's MELD is at below Share15 value, add him to the Share15 List
            elif patient.MELD <Sim.ShareL:
                Share15_2.append(patient)

            #else add him to the Standard List
            else:
                StandardList2.append(patient)

        #iterate patients of the National List
        for patient in NationalList:

            #if patient's MELD is below Share15 value, add him to the Share15 List
            if patient.MELD <Sim.ShareL:
                Share15_3.append(patient)
            #else add him to the second standard list
            else:
                StandardList3.append(patient)

        #sort the lists by MELD Score in decreasing order
        Share35 = sorted(Share35,key=lambda patient: patient.MELD, reverse=True)
        StandardList = sorted(StandardList,key=lambda patient: patient.MELD, reverse=True)
        StandardList2 = sorted(StandardList2,key=lambda patient: patient.MELD, reverse=True)
        StandardList3 = sorted(StandardList3, key=lambda patient:patient.MELD, reverse = True)
        Share15 = sorted(Share15,key=lambda patient: patient.MELD, reverse=True)
        Share15_2 = sorted(Share15_2, key = lambda patient: patient.MELD, reverse = True)
        Share15_3 = sorted(Share15_3, key = lambda patient: patient.MELD, reverse = True)

        #combine the list
        OfferList = Share35 + StandardList + StandardList2 + StandardList3 + Share15 + Share15_2 + Share15_3


    #Execute Match-Run
    if OfferList ==[]:
        return [0,1,[],[]]
    else:
        return MatchRun(organ,OfferList, Sim, Patients_Accept, Donor_Accept, DSA_Avg_Times, AcceptanceModelS1, AcceptanceModel)

    #delete offer list to clear memory
    del OfferList

def MatchRun(offered_organ, offered_list, Sim, Patients_Accept, Donor_Accept, DSA_Avg_Times, AcceptanceModelS1, AcceptanceModel):
    """This function performs a match run for the offered organ on the list of eligible patients.
    It outputs a list containing information on whether the match is found, the DSA of the patient( if match is found),
    and patient id (if match is found)
    Inputs:
        @offered_organ: organ being offered
        @offered_list: list of eligible patients for the offered organ
        @Sim: class object containing relevant variables for simulation
        @Patients_Accept: coefficients regarding patient's characteristics for acceptance model
        @Donor_Accept: coefficients regarding donor's characteristics for acceptance model
        @DSA_Avg_Times: data on average transport time between DSAs
        @AcceptanceModelS1: coefficients regarding patient's characteristics for status-1 acceptance model
        @AcceptanceModel: coefficients regarding patient's characteristics for non-status1 acceptance model
    Outputs:
        @[offerresult[0], offerresult[2], offerresult[3]]: list containing information on whether a match is found
        and patient's information (if match is found)
    """
    noffers =0  #Counts offers made
    offerresult = [0,1,[],[]] #preinitialize result list

    for patient in offered_list:    #Scan offer list
        if MatchCheck(offered_organ, patient, Sim) == 1  and noffers < Sim.maxrejects: #Found matching patient
           offerresult = Offer(offered_organ,patient,noffers, Sim, Patients_Accept, Donor_Accept, DSA_Avg_Times, AcceptanceModelS1, AcceptanceModel)   #make offer
           noffers = noffers + offerresult[1]           #increment number of offers made

           if offerresult[0] == 1:                      #end loop if offer accepted for transplant
               break


        elif noffers >= Sim.maxrejects:                 #end loop if too many offers made
            break

    #return the result vector (match found, matching recipient DSA, matching recipient id)
    return [offerresult[0],offerresult[2],offerresult[3]]


def MatchCheck(offered_organ, potential_recipient, Sim):
    """
    This function peforms an initial check on the offered organ and potential patient to see if patient
    can accept it or not based on biological characteristics
    Input:
        @offered_organ: organ being offered
        @potential_recipient: recipient being checked for eligiblity for the organ
        @Sim: class object containing relevant variables for simulation
    Output:
        @compatible: indicator on whether the patient is eligible for the organ or not
    """
    bcompatible = 0 #initialization for blood compatibility
    active = 1 - potential_recipient.Inactive #if patient is active on the waitlist or not
    ready =1 #default ready indicator

    #if patient is relisted, but his transplant time is before current time, then set ready to 0
    if potential_recipient.Relist ==1 and potential_recipient.RelistTxTime < Sim.clock:
        ready = 0

    #check blood compatibility
    if offered_organ.ABO ==0:
        if potential_recipient.ABO == 0 or potential_recipient.ABO == 1:
                bcompatible = 1
    elif offered_organ.ABO ==1:
        if potential_recipient.ABO == 1:
            bcompatible = 1
    elif offered_organ.ABO ==2:
        if potential_recipient.ABO == 1 or potential_recipient.ABO == 2:
            bcompatible = 1
    else:
            bcompatible = 1

    compatible = bcompatible*active*ready
    return compatible

def Offer(offered_organ, matching_recipient, noffers, Sim, Patients_Accept, Donor_Accept, DSA_Avg_Times, AcceptanceModelS1, AcceptanceModel):
    """This function offers the offered organ to the matching recipient and see if the recipient accepts or not.
    Input:
        @offered_organ: organ being offered
        @matching_recipient: recipient being offered the organ
        @noffers: number of offers made already before the matching recipient
        @Sim: a class object that contain variables relevant to the simulation
        @Patients_Accept: coefficients regarding patient's characteristics for acceptance model
        @Donor_Accept: coefficients regarding donor's characteristics for for acceptance model
        @DSA_Avg_Times: data on average transport time between DSAs
        @AcceptanceModelS1: coefficients regarding patient's characteristics for status-1 acceptance model
        @AcceptanceModel: coefficients regarding patient's characteristics for non-status-1 acceptance model
    Output:
        @[accept, reject, matching_recipient.DSA, matching_recipient.id]: accept is an indicator of whether the patient
        accpets the organ or not, reject is an indicator of whether the patient rejects the organ or not, DSA is the
        patient's DSA, and id is the patient's id
    """
    accept =1 #default acceptance

    #Generate acceptance decision
    r1 = nump.random.uniform(0,1,1)

    #Implement Acceptance Model
    if matching_recipient.Status1 ==1: #Status-1 patient
        patientx = [1,
            float(Patients_Accept[matching_recipient.id][217]),
            float(Patients_Accept[matching_recipient.id][218]),
            float(Donor_Accept[offered_organ.organid][94]),
            DSA_Avg_Times[offered_organ.DSA,matching_recipient.DSA],
            float(Donor_Accept[offered_organ.organid][47]),
            float(Donor_Accept[offered_organ.organid][15]),
            float(Patients_Accept[matching_recipient.id][13]=="True"),
            1,
            0,
            float(float(Patients_Accept[matching_recipient.id][223])<float(Donor_Accept[offered_organ.organid][11])),
            0,
            float(float(Donor_Accept[offered_organ.organid][11])>67.49),
            float(float(Patients_Accept[matching_recipient.id][122])>2),
            float(float(Patients_Accept[matching_recipient.id][122])>2.5)
            ]
        accept_prob = nump.exp(nump.dot(patientx,AcceptanceModelS1)) / (1+nump.exp(nump.dot(patientx,AcceptanceModelS1)))
    else: #non-status 1 patient
        patientx = [1,
            noffers,
            float(Patients_Accept[matching_recipient.id][122]),
            float(Patients_Accept[matching_recipient.id][228]),
            float(Patients_Accept[matching_recipient.id][218]),
            float(Patients_Accept[matching_recipient.id][219]),
            float(Donor_Accept[offered_organ.organid][38]),
            float(Donor_Accept[offered_organ.organid][20]),
            float(matching_recipient.DSA==offered_organ.DSA),
            DSA_Avg_Times[offered_organ.DSA,matching_recipient.DSA],
            (Sim.clock - matching_recipient.create_time)*365,
            float(offered_organ.ABO==2),
            float(offered_organ.ABO==1),
            float(Donor_Accept[offered_organ.organid][98]=="Y"),
            float(Donor_Accept[offered_organ.organid][93]=="P: Positive"),
            float(Donor_Accept[offered_organ.organid][90]=="Y"),
            float(Donor_Accept[offered_organ.organid][96]=="Y"),
            float(Donor_Accept[offered_organ.organid][30]),
            float(Donor_Accept[offered_organ.organid][70]),
            float(Donor_Accept[offered_organ.organid][15]),
            float(Donor_Accept[offered_organ.organid][100]=="128: Native Hawaiian or Other Pacific Islander"),
            float(Donor_Accept[offered_organ.organid][55]),
            float(Donor_Accept[offered_organ.organid][92]=="7: GUNSHOT WOUND"),
            float(Donor_Accept[offered_organ.organid][91]=="6: DEATH FROM NATURAL CAUSES"),
            float(Donor_Accept[offered_organ.organid][95]=="1: NO"),
            float(matching_recipient.ABO==3),
            float(Patients_Accept[matching_recipient.id][13]=="True"),
            float(Patients_Accept[matching_recipient.id][214]=="Y"),
            float(Patients_Accept[matching_recipient.id][212]=="True"),
            float(Patients_Accept[matching_recipient.id][213]=="True"),
            float(Patients_Accept[matching_recipient.id][215]=="Y"),
            float(Patients_Accept[matching_recipient.id][227]=="Y"),
            float(Patients_Accept[matching_recipient.id][64]==Donor_Accept[offered_organ.organid][10]),
            1,
            matching_recipient.lMELD,
            0,
            float(float(Patients_Accept[matching_recipient.id][223])<float(Donor_Accept[offered_organ.organid][11])),
            float(matching_recipient.lMELD>12),
            float(matching_recipient.lMELD>13),
            float(matching_recipient.lMELD>15),
            1,
            float(matching_recipient.MELD>25),
            float(float(matching_recipient.lMELD)+10<float(matching_recipient.MELD)),
            float(float(Donor_Accept[offered_organ.organid][89])>326),
            float(float(Donor_Accept[offered_organ.organid][89])>623),
            float(Donor_Accept[offered_organ.organid][28]),
            float(float(Patients_Accept[matching_recipient.id][216])>3000),
            float(float(Patients_Accept[matching_recipient.id][217])>0),
            float(float(Patients_Accept[matching_recipient.id][222])>154.94),
            float(float(Patients_Accept[matching_recipient.id][122])>.7),
            float(matching_recipient.Na > 131)
            ]
        accept_prob = nump.exp(nump.dot(patientx,AcceptanceModel)) / (1+nump.exp(nump.dot(patientx,AcceptanceModel)))

    #based on acceptance probability, determine to accept or not based on simulation
    accept = int(r1 <= accept_prob)

    #Return information based on decision
    if accept ==1:
        return [1,0,matching_recipient.DSA, matching_recipient.id]

    else:
        return [0,1,[],[]]


##########################################################################################################################################################################