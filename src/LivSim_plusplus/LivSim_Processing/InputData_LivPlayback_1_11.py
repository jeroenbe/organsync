#This module sets up and initializes the input data for LSIM

#import library
from src.LivSim_plusplus.LivSim_Processing import entity
import numpy as nump
import csv
import datetime

def load_data(input_files_location, Sim):

    #########################################################Setting########################################################
    ndsa = 58 #number of DSAs
    i_initial = 1 #Use initial waiting list 1=yes 0=no
    exclude_hi_pr =0 #1=Exclude Hawaii and Puerto Rico
    ########################################################################################################################

    ###################################################Uploading Input Files################################################
    print("Loading file")
    #DSA Geographic Relation Data
    Regions = nump.loadtxt(f"{input_files_location}/Input_Geography.txt")
    Regions = nump.reshape(Regions, (ndsa,ndsa))
    Regions = Regions.astype(int)
    print("Input_Geography.txt loaded.")

    #DSA Sharing Partner Data
    SharingPartners = nump.loadtxt(f"{input_files_location}/Input_SPartners2.txt")
    SharingPartners = nump.reshape(SharingPartners, (ndsa,ndsa))
    SharingPartners = SharingPartners.astype(int)
    print("Input_SPartners.txt loaded.")

    #Patient Arrival Input Data
    PatientPlayback = nump.loadtxt(f"{input_files_location}/Patients.txt")
    print("Patients.txt loaded.")

    #Organ Input Data
    OrganPlayback = nump.loadtxt(f"{input_files_location}/Donors.txt")
    print("Donors.txt loaded.")

    #Progession Input Data
    ProgressionPlayback = nump.loadtxt(f"{input_files_location}/Status.txt")
    print("Status.txt loaded.")

    #Relist Data
    Relist = nump.loadtxt(f"{input_files_location}/Input_Relist.txt")
    print("Input_Relist.txt loaded.")

    #Acceptance Model Data
    AcceptanceModel = nump.loadtxt(f"{input_files_location}/Input_Acceptance.txt")
    print("Input_Acceptance.txt loaded.")

    AcceptanceModelS1 = nump.loadtxt(f"{input_files_location}/Input_Acceptance_Status1.txt")
    print("Input_Acceptance_Status1.txt loaded.")

    DSA_Avg_Times = nump.loadtxt(f"{input_files_location}/DSA_AvgTimes.txt")
    DSA_Avg_Times  = nump.reshape(DSA_Avg_Times , (ndsa,ndsa))
    DSA_Avg_Times = DSA_Avg_Times.astype(float)
    print("DSA_AvgTimes.txt loaded.")

    #Donor Accept data which has contains organs that become available from the model start date through and including the model end date
    with open(f"{input_files_location}/Donor_Accept.txt",'r') as inputf:
        reader = list(csv.reader(inputf, delimiter = '|'))
        Donor_Accept = []
        for row in reader:
            Donor_Accept.append(row)
    print("Donor_Accept.txt loaded.")

    #Patient Accept data which contains all candidates on the waiting list as of the day befor the model start and new patinets
    #added to the waiting list with listing dats from the model start date through and including the model end date
    with open(f"{input_files_location}/Patients_Accept.txt",'r') as inputf:
        reader = list(csv.reader(inputf, delimiter = '|'))
        Patients_Accept = []
        for row in reader:
            Patients_Accept.append(row)
    print("Patients_Accept.txt loaded.")

    # Initial Waiting List
    OPTN_initial = [[] for i in range(0,ndsa)]
    initial_counts = 0*nump.ndarray(shape=(ndsa,1), dtype=int)
    if i_initial ==1:
        #upload initial waitlist of patients waiting for transplant
        InitialList = nump.loadtxt(f"{input_files_location}/Waitlist_matchmeld.txt")
        initialrows = nump.shape(InitialList)[0]

        #Based on inclusion of Hawaii and Puerto Rico, select appropriate column for DSA ids
        if exclude_hi_pr ==1:
            dsa_id_column = 1
        else:
            dsa_id_column =8

        #init_prog_scheduler = []

        #iterate through list of patients of intial waitlist
        for i in range(0, initialrows):
            if InitialList[i,dsa_id_column] > 56 and exclude_hi_pr ==1: #Excluding Hawaii and Puerto Rico
                pass
            else:
                newpatient = entity.Patient(InitialList[i,0].astype(int),InitialList[i,dsa_id_column].astype(int),InitialList[i,2]) #construct patient object
                
                #record patient information in patient object
                newpatient.ABO = InitialList[i,3].astype(int)
                newpatient.Status1 = InitialList[i,6].astype(int)
                newpatient.Inactive = InitialList[i,9].astype(int)
                if InitialList[i,7] != ".":
                    newpatient.Na = InitialList[i,7].astype(int)
                else:
                    newpatient.Na = 137
                labmeld = InitialList[i,4].astype(int)

                #Assign MELD to Status1 HCC candidates
                if newpatient.Status1 ==1:
                    newpatient.lMELD =41
                    newpatient.MELD =41


                #Assign MELD and HCC for non-Status 1 candidates
                if newpatient.Status1 ==0:
                    newpatient.HCC =InitialList[i,5].astype(int)

                #if MELD sodium option is selected, compute MELD for non-HCC, non-Status1 patient based on sodium score
                if Sim.sodium == 1 and newpatient.Status1 ==0 and newpatient.HCC ==0:
                    effective_na = newpatient.Na

                    #set bound on sodium level
                    if effective_na <125:
                        effective_na = 125
                    elif effective_na > 137:
                        effective_na = 137

                    #set lab meld score
                    newpatient.lMELD = labmeld

                    #compute the allocation meld score
                    newpatient.MELD = nump.rint(labmeld + 1.32*(137-effective_na)-(0.033*labmeld*(137-effective_na)))

                #if MELD sodium is option is not selected for non-HCC, non-Status1 patient, set patient's MELD to lab meld score
                elif Sim.sodium == 0 and newpatient.Status1 ==0 and newpatient.HCC ==0:
                    newpatient.lMELD = labmeld
                    newpatient.MELD = labmeld

                #if patient is HCC, adjust MELD score
                elif newpatient.HCC ==1:
                    #If cap and delay not selected, adjust MELD score as follows.
                    if Sim.capanddelay ==0:
                                if labmeld <=22:
                                    newpatient.MELD =22
                                elif labmeld <=25 and labmeld > 22:
                                    newpatient.MELD =25
                                elif labmeld <=28 and labmeld > 25:
                                    newpatient.MELD =28
                                elif labmeld <=29 and labmeld > 28:
                                    newpatient.MELD =29
                                elif labmeld <=31 and labmeld > 29:
                                    newpatient.MELD =31
                                elif labmeld <=33 and labmeld > 31:
                                    newpatient.MELD =33
                                elif labmeld <=34 and labmeld > 33:
                                    newpatient.MELD =34
                                elif labmeld <=35 and labmeld > 34:
                                    newpatient.MELD =35
                                elif labmeld <=37 and labmeld > 35:
                                    newpatient.MELD =37
                                elif labmeld > 37:
                                    newpatient.MELD =min(labmeld,40)
                    #if cap and delay is selected, adjust MELD score based on how long the patient waited by the start of the
                    #model
                    else:
                                    if (0- newpatient.create_time) <= .5:
                                        newpatient.MELD = 28
                                    elif (0 - newpatient.create_time) > .5 and (0 - newpatient.create_time <= .75):
                                        newpatient.MELD = 29
                                    elif (0 - newpatient.create_time) > .75 and (0 - newpatient.create_time <= 1):
                                        newpatient.MELD = 31
                                    elif (0 - newpatient.create_time) > 1 and (0 - newpatient.create_time <= 1.25):
                                        newpatient.MELD = 33
                                    elif (0 - newpatient.create_time) > 1.25 and (0 - newpatient.create_time <= 1.5):
                                        newpatient.MELD = 34
                                    else:
                                        newpatient.MELD = min(labmeld,40)
                    #set lab meld score to allocation meld score
                    newpatient.lMELD =newpatient.MELD


            #if newpatient.HCC ==1 and Sim.capanddelay ==0:
            #   init_prog_scheduler.append([.25, newpatient.DSA,newpatient.id])
            #elif newpatient.HCC ==1 and Sim.capanddelay ==1 and -newpatient.create_time <=.25:
            #   init_prog_scheduler.append([.5, newpatient.DSA,newpatient.id])
            #elif newpatient.HCC ==1 and Sim.capanddelay ==1 and -newpatient.create_time >.25:
            #   init_prog_scheduler.append([.25, newpatient.DSA,newpatient.id])
            #else:
            #   init_prog_scheduler.append([Sim.progtime, newpatient.DSA,newpatient.id])

            #record the number of patients in waiting list for each DSA
            initial_counts[newpatient.DSA] = initial_counts[newpatient.DSA] + 1

            #add the patient to the corresponding DSA waiting list
            if newpatient.DSA >=0:
                OPTN_initial[newpatient.DSA].append(newpatient)
    print("Waitlist_matchmeld.txt loaded.")



    print("Simulation Input Complete. Starting simulation @ " ,datetime.datetime.now().time())

    ########################################################################################################################

    return ndsa, i_initial, exclude_hi_pr, Regions, SharingPartners, PatientPlayback, OrganPlayback, ProgressionPlayback, Relist, AcceptanceModel, AcceptanceModelS1, DSA_Avg_Times, Donor_Accept, Patients_Accept, OPTN_initial, initial_counts