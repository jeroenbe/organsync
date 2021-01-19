from src.LivSim_plusplus.LivSim_Processing import entity, allocate
import numpy as nump
import datetime
from copy import deepcopy
ndsa = 58

#######################################################################Event Processes####################################################################################
def Arrival(arrivalinfo, Sim, Stat, OPTN):
    """
    This function simulates the arrival of patients. It computes the MELD score of the patient and adds him to the corresponding DSA waiting list.
    Input:
        @arrivalinfo: all the patient's information
        @Sim: class object that contains variables relevant to the simulation
        @Stat: class object that containts statistical info of the simulation
        @OPTN: complete patient data
    Output:
        @Sim: updated Sim
        @Stat: updated Stat
        @OPTN updated OPTN
    """
    #Create Patient Entity
    newpatient = entity.Patient(arrivalinfo[1].astype(int),arrivalinfo[3].astype(int),Sim.clock)

    #Assign Patient Characteristics
    newpatient.Status1 = arrivalinfo[9].astype(int)
    newpatient.ABO = arrivalinfo[5].astype(int)
    newpatient.HCC = arrivalinfo[8].astype(int)
    newpatient.Na = arrivalinfo[10].astype(int)
    newpatient.lMELD = arrivalinfo[7].astype(int)
    newpatient.MELD = arrivalinfo[6].astype(int)
    newpatient.Inactive = arrivalinfo[11].astype(int)
    
    #Assign Allocation MELD based on policy
    if Sim.sodium ==1: #if sodium policy is selected
        
        #bound the sodium score
        effective_na = newpatient.Na
        if effective_na <125:
            effective_na = 125
        elif effective_na > 137:
            effective_na = 137

        #compute the allocation MELD score
        if nump.rint(newpatient.lMELD + 1.32*(137-effective_na)-(0.033*newpatient.lMELD*(137-effective_na))) <6:
            newpatient.MELD =6
        elif nump.rint(newpatient.lMELD + 1.32*(137-effective_na)-(0.033*newpatient.lMELD*(137-effective_na))) > 40:
            newpatient.MELD =40
        else:
            newpatient.MELD = nump.rint(newpatient.lMELD + 1.32*(137-effective_na)-(0.033*newpatient.lMELD*(137-effective_na)))
    
    else: #if sodium policy not selected

        #bound the sodium score
        if newpatient.MELD <6:
            newpatient.MELD =6
        elif newpatient.MELD >40:
            newpatient.MELD = 40


    #Apply Status1 and HCC exceptions (if applicable)
    if newpatient.Status1 ==1:
        newpatient.MELD =41
    elif newpatient.HCC ==1 and Sim.capanddelay ==1:
        newpatient.MELD = min(newpatient.lMELD,28)

    #Place Patient in DSA List
    if newpatient.DSA >= 0:
        OPTN[newpatient.DSA].append(newpatient)


    #Update Stats
    Stat.yarrivals[newpatient.DSA] = Stat.yarrivals[newpatient.DSA] + 1
    Stat.numcandidates[newpatient.DSA] = Stat.numcandidates[newpatient.DSA] + 1

    #return updated Sim, Stat, and OPTN
    return Sim, Stat, OPTN


    #Diagnostic Info
    #here

def Progression(proginfo, Sim, Stat, OPTN, reps):
    """
    This function searches for a particular patient from the OPTN data structure and update the patient's characteristics
    Input:
        @proginfo: all of the information of the patient being searched up
        @Sim: class object containing relevant information of the patient being searched up
        @Stat: class object containing statistical information of the simulation
        @OPTN: complete patient data
        @reps: current replication number
    Output:
        @Sim: updated Sim
        @Stat: updated Stat
        @OPTN: updated OPTN
    """
    progdsa = proginfo[9].astype(int) #obtain DSA
    #test =0

    #run if we have an actual DSA
    if progdsa >=0:

        #search for patient in the OPTN data structure
        for i, patient in enumerate(OPTN[progdsa]):

            #check if the id matches
            if patient.id == proginfo[1].astype(int):
                
                #Update Patient
                if patient.Relist ==1:
                    
                    #Ignore updates for relisted patients
                    break
                

                elif proginfo[3].astype(int) == 1:
                    
                    #Patient dies, remove and update stats
                    Stat.ydeaths[progdsa] = Stat.ydeaths[progdsa] + 1 #increment the number of deaths by one
                    Stat.numcandidates[progdsa] = Stat.numcandidates[progdsa] - 1 #decrement the number of candidates waiting by one
                    del OPTN[progdsa][i] #delete the patient object since patient died
                    break

                elif proginfo[4].astype(int) ==1:
                    
                    #Patient removed, remove and update stats
                    Stat.yremoved[progdsa] = Stat.yremoved[progdsa] + 1 #increment the number of removals by one
                    Stat.numcandidates[progdsa] = Stat.numcandidates[progdsa] - 1 #decrement the number of candidates waiting by one
                    
                    #record as follows (time of removal, repetition, patient id, patient allocation MELD, patient lab MELD)
                    oidreport = [nump.floor(Sim.clock), reps, Sim.clock, patient.id,patient.MELD, patient.lMELD] 
                    Sim.record_removals = nump.vstack((Sim.record_removals,  oidreport)) #concatenate the row oidreport to the record of removals
                    del OPTN[progdsa][i] #delete patient object since patient is removed
                    break

                else:

                    #Update candidate
                    patient.lMELD = proginfo[6].astype(int)
                    patient.Na = proginfo[7].astype(int)
                    patient.MELD = proginfo[5].astype(int)
                    patient.Inactive = proginfo[10].astype(int)

                    #set bound on MELD score
                    if patient.MELD <6:
                        patient.MELD =6
                    elif patient.MELD >40:
                        patient.MELD = 40

                    #Update Allocation MELD based on policy (if applicable)
                    if Sim.sodium ==1 and patient.Status1 != 1 and patient.HCC != 1:
                        #if sodium score policy is selected, then update the meld score for non-status1, non-HCC patient

                        #set bound on sodium score
                        effective_na = patient.Na
                        if effective_na <125:
                            effective_na = 125
                        elif effective_na > 137:
                            effective_na = 137

                        #compute MELD score
                        if nump.rint(patient.lMELD + 1.32*(137-effective_na)-(0.033*patient.lMELD*(137-effective_na))) <6:
                            patient.MELD =6
                        elif nump.rint(patient.lMELD + 1.32*(137-effective_na)-(0.033*patient.lMELD*(137-effective_na))) > 40:
                            patient.MELD =40
                        else:
                            patient.MELD = nump.rint(patient.lMELD + 1.32*(137-effective_na)-(0.033*patient.lMELD*(137-effective_na)))

                    elif Sim.capanddelay ==1 and patient.Status1 != 1 and patient.HCC == 1:
                        #if cap and delay policy is selected, update meld score for status1, HCC patient

                            #compute MELD score
                            if Sim.clock - patient.create_time <= .5:
                                patient.MELD = max(28,patient.MELD)
                            elif (Sim.clock - patient.create_time) > .5 and (Sim.clock - patient.create_time <= .75):
                                patient.MELD = max(29,patient.MELD)
                            elif (Sim.clock - patient.create_time) > .75 and (Sim.clock - patient.create_time <= 1):
                                patient.MELD = max(31,patient.MELD)
                            elif (Sim.clock - patient.create_time) > 1 and (Sim.clock - patient.create_time <= 1.25):
                                patient.MELD = max(33,patient.MELD)
                            elif (Sim.clock - patient.create_time) > 1.25 and (Sim.clock - patient.create_time <= 1.5):
                                patient.MELD = max(34,patient.MELD)
                            else:
                                patient.MELD = min(patient.MELD+1,40)
                    break
    #return updated Sim, Stat, and OPTN
    return Sim, Stat, OPTN




def OrganArrival(organinfo, Sim, Stat, OPTN, Regions, SharingPartners, Patients_Accept, Donor_Accept, DSA_Avg_Times, AcceptanceModelS1, AcceptanceModel, Relist, reps):
    """
    This function simulates the organ arrival. It tries to match the organ to a patient from the corresponding DSA waitlist.
    Input:
        @organinfo: information on the organ
        @Sim: class object containing relevant variables for simulation
        @Stat: class object containing statistical information of simulation
        @OPTN: complete patient data
        @Regions: neighbhorhood map for regions, districts, or neighbhorhoods
        @SharingPartners: neighborhood map adding sharing partners to existing geographic relationships among OPOs
        @Patients_Accept: coefficients regarding donor's characteristics for acceptance model
        @Donor_Accept: coefficients regarding donor's characteristics for acceptance model
        @DSA_Avg_Times: data on average transport times between DSAs
        @AcceptanceModelS1: coefficients regarding patient's characteristics for status-1 acceptance model
        @AccpetanceModel: coefficients regarding patient's characteristics for non-status-1 acceptance model
        @Relist: values regarding the probability that a transplanted patient will relist
        @reps: replication number
    Output:
        @Sim: updated Sim
        @Stat: updated Stat
        @OPTN: updated OPTN
    """

    #Create Organ
    neworgan = entity.Organ(int(organinfo[2]))
    neworgan.organid = Sim.oid
    Sim.oid = Sim.oid + 1

    #Assign Organ Attributes
    neworgan.ABO = organinfo[4].astype(int)


    #Allocate Organ
    #disposition is tuple of organ status (accept/reject) , transplanting DSA, and patient id if accepted
    disposition = allocate.Allocate(neworgan, OPTN, Sim, Regions, SharingPartners, Patients_Accept, Donor_Accept, DSA_Avg_Times, AcceptanceModelS1, AcceptanceModel)

    if disposition[0] == 1: #organ is transplanted

        #Remove transplanted patient from waiting list and update statistics
        for i, patient in enumerate(OPTN[disposition[1]]):

            #search for the patient in the OPTN data structure
            if patient.id == disposition[2]:

                #Determine if patient will relist and assign special attributes if necesary
                willrelist = 0

                if patient.Relist ==0:
                #if patient has not been relisted

                    #Determine if patient will be relisted if was not already
                    r1 =nump.random.uniform(Relist[0],Relist[1],1)
                    r2 =nump.random.uniform(0,1,1)

                    if r2 < r1:
                        willrelist =1
                        
                        #Determine when current graft will fail
                        r3 = nump.random.uniform(0,1,1)
                        if r3 < .4:
                            patient.RelistTxTime = Sim.clock + 5
                        elif r3 >= .4 and r3 < .6:
                            patient.RelistTxTime = Sim.clock + 2
                        elif r3 >= 6 and r3 < .8:
                            patient.RelistTxTime = Sim.clock + 1
                        else:
                            patient.RelistTxTime = Sim.clock

                        #Update relist statistics
                        Stat.yrelists[disposition[1]] = Stat.yrelists[disposition[1]] +1

                        #record the floor of current time, reptitiion, current time, patient id, patient meld score, and relist tx time
                        relistidreport = [nump.floor(Sim.clock), reps, Sim.clock, patient.id,patient.MELD,patient.RelistTxTime]

                        #concatenate the relistidreport to the record of relists
                        Sim.record_relists = nump.vstack((Sim.record_relists,  relistidreport))

                #Update Stats for Transplants
                #Number of Transplants and Regrafts
                Stat.ytransplants[disposition[1]] = Stat.ytransplants[disposition[1]] +1
                Sim.record_txDSA[neworgan.DSA,disposition[1]] = Sim.record_txDSA[neworgan.DSA,disposition[1]] +1
                
                if patient.Relist ==1:
                #if patient is relisted
                    Stat.yregrafts[disposition[1]] = Stat.yregrafts[disposition[1]] +1 #increase number of retransplanted patients by 1

                #Compute waiting Time
                if patient.Relist ==0:
                    #if patient is not relisted
                    Stat.ywait[disposition[1]] = Stat.ywait[disposition[1]] + (Sim.clock - patient.create_time)
                else:
                    #if patient is relisted
                    Stat.ywait[disposition[1]] = Stat.ywait[disposition[1]] + (Sim.clock - patient.RelistTxTime)

                #Waiting List Sizes
                if willrelist ==0:
                    Stat.numcandidates[disposition[1]] = Stat.numcandidates[disposition[1]] -1 #decrease the number of waitling list candidates for the DSA by 1

                #1st Transplant MELD
                if patient.Status1 ==0 and patient.Relist ==0:
                    #Tx-MELD at measure assumed to exclude re-grafts
                    Stat.yMELD[disposition[1]] = Stat.yMELD[disposition[1]] + patient.MELD #increase the MELD score the DSA
                    Stat.ymedMELD[disposition[1]].append(patient.MELD) #record the patient MELD score

                #Output for Posttransplant processing for those who were not ever relisted or will be
                if willrelist ==0 and patient.Relist ==0:
                    regtx =0
                    nattx = 0
                    if patient.DSA != neworgan.DSA and (Regions[neworgan.DSA,patient.DSA] ==1 or SharingPartners[neworgan.DSA, patient.DSA] == 1):
                        regtx =1 #patient had regional transplant
                    elif patient.DSA != neworgan.DSA:
                        nattx=1 #patient had national transplant

                    #record the floor of the current time, repetition, current time, patient id, indicator for regional transplant, indicator for national
                    #transplant
                    txidreport = [nump.floor(Sim.clock), reps, Sim.clock, patient.id,regtx,nattx]
                    Sim.record_txID = nump.vstack((Sim.record_txID,  txidreport)) #add new record to list of transplant records

                    #record as follows (time of removal, repetition, patient id, patient allocation MELD, patient lab MElD)
                    oidreport = [nump.floor(Sim.clock), reps, Sim.clock, patient.id,Sim.oid]

                    #add to list of transplant records
                    Sim.record_doID = nump.vstack((Sim.record_doID,  oidreport))

                #Out for Posttransplant proceesing for regrafts
                if patient.Relist ==1:
                    regtx =0 #indicator for regional transplant
                    nattx = 0 #indicator for national transplant

                    if patient.DSA != neworgan.DSA and (Regions[neworgan.DSA,patient.DSA] ==1 or SharingPartners[neworgan.DSA, patient.DSA] == 1):
                        regtx =1 #patient had regional transplant
                    elif patient.DSA != neworgan.DSA:
                        nattx=1 #patient had national transplant

                    #record the floor of the current time, repetition, current time, patient id, indicator for regional transplant, indicator for national transplant
                    txidreport = [nump.floor(Sim.clock), reps, Sim.clock, patient.id,regtx,nattx]

                    #add to list of relisted transplants
                    Sim.record_txIDregraft = nump.vstack((Sim.record_txIDregraft,  txidreport))

                    #record as follows (time of removal, repetition, patient id, patient allocation MELD, patient lab MELD)
                    oidreport = [nump.floor(Sim.clock), reps, Sim.clock, patient.id,Sim.oid]

                    #add to list of retransplant records
                    Sim.record_doIDregraft = nump.vstack((Sim.record_doIDregraft,  oidreport))


                
                if willrelist ==1:
                    #if patient will relist, update relist status and MELD score
                    OPTN[disposition[1]][i].Relist =1
                    OPTN[disposition[1]][i].MELD = 32
                    OPTN[disposition[1]][i].lMELD = 32
                else:
                    #remove transplanted patient if will not be relisted
                    del OPTN[disposition[1]][i]
                break

    else: #organ is discarded; update statistics (optional)
        pass

    #return updated Sim, Stat, and OPTN
    return Sim, Stat, OPTN

def Year(Sim, Stat, reps):
    """
    This function updates the statistics of the simulation per year
    Input:
        @Sim: class object that contains relevant variables for the simulation
        @Stat: class object that contains statistical information of the simulation
        @reps: current replication number
    Output:
        @Sim: updated Sim
        @Stat: updated Stat
    """

    #Annual Disparity Statistics
    mr_1 = nump.zeros(shape=(ndsa,1),dtype=float)
    tr_1 = nump.zeros(shape=(ndsa,1),dtype=float)
    wt_1 = nump.zeros(shape=(ndsa,1),dtype=float)
    meld_l = nump.zeros(shape=(ndsa,1),dtype=float)

    for i in range(0,ndsa):
        if Stat.ytransplants[i] > 0:
            wt_1[i] = Stat.ywait[i] / Stat.ytransplants[i] #compute the total waiting list/total # of transplant per DSA
            meld_l[i] = Stat.yMELD[i] / Stat.ytransplants[i] #compute the total MELD score/total # of transplant per DSA
        else:
            #write nan if no values available
            wt_1[i] = nump.nan
            meld_l[i] = nump.nan


        if (Stat.yarrivals[i] + Stat.ycandidates[i]) == 0:
            #write nan if no values available
            mr_1[i] = nump.nan
            tr_1[i] = nump.nan
        else:
            mr_1[i] = Stat.ydeaths[i] / (Stat.yarrivals[i] + Stat.ycandidates[i]) #compute mortality rate (number of deaths/number of waiting candidates)
            tr_1[i] = Stat.ytransplants[i] / (Stat.yarrivals[i] + Stat.ycandidates[i]) #compute transplant rate (number of transplants/number of waiting candidates)

    #compute the median MELD score
    medianmelds = nump.zeros(shape=(ndsa,1),dtype=float)
    for i in range(0,ndsa):
        if Stat.ymedMELD[i] != []:
            medianmelds[i] = nump.nanmedian(Stat.ymedMELD[i])
        else:
            medianmelds[i] = nump.nan


    #Intermediate Data Outputs
    #nump.savetxt("Output_check.txt", Stat.numcandidates, fmt='%1.4e', delimiter='\t', newline='\n')
    #nump.savetxt("Output_check2.txt", Stat.yremoved, fmt='%1.4e', delimiter='\t', newline='\n')
    #nump.savetxt("Output_check3.txt", Stat.yarrivals, fmt='%1.4e', delimiter='\t', newline='\n')
    #nump.savetxt("Output_check4.txt", Stat.ydeaths, fmt='%1.4e', delimiter='\t', newline='\n')
    #nump.savetxt("Output_check5.txt", Stat.ytransplants, fmt='%1.4e', delimiter='\t', newline='\n')
    #nump.savetxt("Output_check6.txt", mr_1, fmt='%1.4e', delimiter='\t', newline='\n')



    mr_numdeaths = [nump.sum(Stat.ydeaths),nump.floor(Sim.clock),reps] #record the total number of deaths along with its current time and current repetition
    #mr_disparity = [nump.linalg.norm(mr_1,ord=1),nump.floor(Sim.clock),reps] 
    #tx_disparity = [nump.linalg.norm(tr_1,ord=1),nump.floor(Sim.clock),reps]
    #wt_disparity = [nump.linalg.norm(wt_1,ord=1),nump.floor(Sim.clock),reps]
    mr_disparity_mean = [nump.nanmean(mr_1),nump.floor(Sim.clock),reps] #record the mean mortality rate along with its current time and current repetition
    mr_disparity_std = [nump.nanstd(mr_1),nump.floor(Sim.clock),reps] #record the standard deviation of mortality rate along withs current time and current repetition
    meld_disparity_mean = [nump.nanmean(meld_l),nump.floor(Sim.clock),reps] #record the mean MELD score along with current time and current repetition
    meld_disparity_std = [nump.nanstd(meld_l),nump.floor(Sim.clock),reps] #record the standard deviation of the MELD score along with current time and current repetition
    medmeld_mean = [nump.nanmean(medianmelds),nump.floor(Sim.clock),reps] #record the mean median MELD score along with current time and current repetition
    medmeld_std = [nump.nanstd(medianmelds),nump.floor(Sim.clock),reps] #record the standard deviation of the median MELD score along with current time and current repetition

    #print(tx_disparity)

    #add the records to the list of yearly statistics
    Sim.record_deaths = nump.vstack((Sim.record_deaths,  mr_numdeaths))
    Sim.record_mr_disparity_mean = nump.vstack((Sim.record_mr_disparity_mean, mr_disparity_mean))
    Sim.record_mr_disparity_std = nump.vstack((Sim.record_mr_disparity_std, mr_disparity_std))
    Sim.record_meld_disparity_mean = nump.vstack((Sim.record_meld_disparity_mean, meld_disparity_mean))
    Sim.record_meld_disparity_std = nump.vstack((Sim.record_meld_disparity_std, meld_disparity_std))
    Sim.record_medMELDmean = nump.vstack((Sim.record_medMELDmean, medmeld_mean))
    Sim.record_medMELDstd = nump.vstack((Sim.record_medMELDstd, medmeld_std))
    Sim.record_txDSAoutput = nump.vstack((Sim.record_txDSAoutput, Sim.record_txDSA))

    #create array that records the current time and repetition
    recindex =nump.ndarray(shape=(1,3))
    recindex[0,0] = nump.floor(Sim.clock)
    recindex[0,1] = reps

    #add DSA vector regarding deaths, transplants, etc. to the list of records
    Sim.record_ydeaths =nump.concatenate((Sim.record_ydeaths,nump.concatenate((recindex,nump.transpose(Stat.ydeaths)),axis=1)),axis=0)
    Sim.record_ytransplants = nump.concatenate((Sim.record_ytransplants,nump.concatenate((recindex,nump.transpose(Stat.ytransplants)),axis=1)),axis=0)
    Sim.record_yarrivals = nump.concatenate((Sim.record_yarrivals,nump.concatenate((recindex,nump.transpose(Stat.yarrivals)),axis=1)),axis=0)
    Sim.record_ycandidates = nump.concatenate((Sim.record_ycandidates,nump.concatenate((recindex,nump.transpose(Stat.ycandidates)),axis=1)),axis=0)
    Sim.record_yremoved =nump.concatenate((Sim.record_yremoved,nump.concatenate((recindex,nump.transpose(Stat.yremoved)),axis=1)),axis=0)
    Sim.record_ywait = nump.concatenate((Sim.record_ywait,nump.concatenate((recindex,nump.transpose(Stat.ywait)),axis=1)),axis=0)
    Sim.record_yMELD = nump.concatenate((Sim.record_yMELD,nump.concatenate((recindex,nump.transpose(Stat.yMELD)),axis=1)),axis=0)
    Sim.record_yrelists =nump.concatenate((Sim.record_yrelists,nump.concatenate((recindex,nump.transpose(Stat.yrelists)),axis=1)),axis=0)
    Sim.record_yregrafts =nump.concatenate((Sim.record_yregrafts,nump.concatenate((recindex,nump.transpose(Stat.yregrafts)),axis=1)),axis=0)

    #Reset Statistics for Following Year
    Stat.yarrivals =    nump.zeros(shape=(ndsa,1),dtype=int)
    Stat.ydeaths =      nump.zeros(shape=(ndsa,1),dtype=int)
    Stat.yremoved =     nump.zeros(shape=(ndsa,1),dtype=int)
    Stat.ytransplants = nump.zeros(shape=(ndsa,1),dtype=int)
    Stat.ycandidates = deepcopy(Stat.numcandidates)
    Stat.ywait =        nump.zeros(shape=(ndsa,1),dtype=float)
    Stat.yMELD =        nump.zeros(shape=(ndsa,1),dtype=int)
    Stat.ymedMELD =     [[] for i in range(0,ndsa)]
    Stat.yrelists =     nump.zeros(shape=(ndsa,1),dtype=int)
    Stat.yregrafts =    nump.zeros(shape=(ndsa,1),dtype=int)

    #return updated Sim, Stat
    return Sim, Stat



def EndRep():
    """
    Prints a message saying that a replication ended.
    """
    print("Ending replication,  time is: ", datetime.datetime.now().time())



    #print(Sim.clock)

#####################################################################################################################################################################################################################