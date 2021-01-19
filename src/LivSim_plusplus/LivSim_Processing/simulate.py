#!/sscc/opt/anaconda3/bin/python
import heapq
import numpy as nump
import scipy as scip
import datetime
import operator
import sys
import os
import queue
import csv
import ast
import pandas as pd
from copy import deepcopy

from src.LivSim_plusplus.LivSim_Processing import engine, entity, allocate, event
from src.LivSim_plusplus.LivSim_Processing.InputData_LivPlayback_1_11 import load_data

import click



# maxtime = float(sys.argv[1])
# nreps = int(sys.argv[2])
# policy = ast.literal_eval(sys.argv[3])
# print('policy', policy)
# ShareU = ast.literal_eval(sys.argv[4])
# ShareL = ast.literal_eval(sys.argv[5])

# localboost = int(sys.argv[6])

# directory = sys.argv[7]

@click.command()
@click.argument('maxtime', type=int)
@click.argument('nreps', type=int)
@click.argument('policy', type=int, nargs=4)
@click.argument('share_u', type=int)
@click.argument('share_l', type=int)
@click.argument('localboost', type=int)
@click.argument('directory', type=click.Path(exists=True))
@click.option('--input_files_location', type=click.Path(exists=True), default='./src/LivSim_plusplus/demo_input')
def cli(maxtime, nreps, policy, share_u, share_l, localboost, directory, input_files_location):
    ShareU, ShareL = share_u, share_l

    Sim = engine.G()
    nump.random.seed(Sim.seed)
    
    ndsa, i_initial, exclude_hi_pr, Regions, SharingPartners, PatientPlayback, OrganPlayback, ProgressionPlayback, Relist, AcceptanceModel, AcceptanceModelS1, DSA_Avg_Times, Donor_Accept, Patients_Accept, OPTN_initial, initial_counts = load_data(input_files_location, Sim)
    
    for reps in range(1,nreps+1):
        #Initialize Replication
        Sim.clock=0
        #Sim.pid = 100000
        Sim.oid = 0	

        #Update parameters
        Sim.maxtime = maxtime
        Sim.nreps = nreps
        Sim.regional_sharing = policy[0]
        Sim.sodium = policy[1]
        Sim.capanddelay = policy[2]
        Sim.spartners = policy[3]
        Sim.ShareU = ShareU
        Sim.ShareL = ShareL
        Sim.localboost = localboost

        #Initialize Statistics
        Stat = engine.SimStat()
        Stat.yarrivals =    nump.zeros(shape=(ndsa,1),dtype=int)
        Stat.ydeaths =      nump.zeros(shape=(ndsa,1),dtype=int)
        Stat.yremoved =     nump.zeros(shape=(ndsa,1),dtype=int)
        Stat.ytransplants = nump.zeros(shape=(ndsa,1),dtype=int)
        Stat.ywait = nump.zeros(shape=(ndsa,1),dtype=float)
        Stat.yMELD = nump.zeros(shape=(ndsa,1),dtype=int)
        Stat.numcandidates =  deepcopy(initial_counts)
        Stat.ycandidates =    deepcopy(Stat.numcandidates)
        Stat.ymedMELD = [[] for i in range(0,ndsa)]
        Stat.yrelists =     nump.zeros(shape=(ndsa,1),dtype=int)
        Stat.yregrafts =    nump.zeros(shape=(ndsa,1),dtype=int)
        print("Starting replication,  time is: ", datetime.datetime.now().time())	

        #Initialize Waiting Lists
        OPTN = deepcopy(OPTN_initial)	
    

        #Schedule events for playback
        scheduling_done =0
        nextyear =1
        Calendar = []	

        #extract subset relted to the replications
        subset_PatientPlayback = PatientPlayback[PatientPlayback[:,0] == reps]
        subset_OrganPlayback = OrganPlayback[OrganPlayback[:,0] == reps]
        subset_ProgressionPlayback = ProgressionPlayback[ProgressionPlayback[:,0] == reps]	

        #set up index pointers
        patient_index_pointer = 0
        organ_index_pointer = 0
        prog_index_pointer = 0	

        #create indices
        patient_index = nump.size(subset_PatientPlayback,0)-1
        organ_index = nump.size(subset_OrganPlayback,0)-1
        prog_index = nump.size(subset_ProgressionPlayback,0)-1	
    

        while scheduling_done ==0:	

            #next patient arrival
            if patient_index_pointer <= patient_index:
                nextarrival = subset_PatientPlayback[patient_index_pointer][4] #next arrival is next patient's arrival time
            else:
                nextarrival = Sim.maxtime +1 #next arrival is 1+Sim.maxtime	

            #next organ arrival
            if organ_index_pointer <= organ_index:
                nextorgan = subset_OrganPlayback[organ_index_pointer][3] #next organ arrival is next organ's arrival time
            else:
                nextorgan = Sim.maxtime +1 #next organ arrival is 1+Sim.maxtime	

            #next progression arrival
            if prog_index_pointer <= prog_index:	

                nextprog = subset_ProgressionPlayback[prog_index_pointer][2] #next progression is next status event time
            else:
                nextprog = Sim.maxtime +1 #next progression is 1+Sim.maxtime	
    
    

            if nextyear == min(nextyear,nextarrival,nextorgan,nextprog,Sim.maxtime):
            #if the next event is the next year, then record event as a Year and add to calendar
                nextevent = engine.Event('Year',nextyear,[])
                Calendar.append(nextevent)
                nextyear = nextyear + 1	

            elif nextarrival == min(nextarrival,nextorgan,nextprog,Sim.maxtime):
            #if the next event is a patient arrival, record event as a patient arrival and add to calendar; update patient arrival pointer
                nextevent = engine.Event('Arrival',nextarrival,subset_PatientPlayback[patient_index_pointer])
                Calendar.append(nextevent)
                patient_index_pointer = patient_index_pointer + 1	

            elif nextorgan == min(nextorgan,nextprog,Sim.maxtime):
            #if the next event is organ arrival, record event as organ arrival and add to calendar; update organ arrival pointer
                nextevent = engine.Event('Organ',nextorgan,subset_OrganPlayback[organ_index_pointer])
                Calendar.append(nextevent)
                organ_index_pointer = organ_index_pointer + 1	

            elif nextprog <= Sim.maxtime:
            #if next event is progression, record event as progression and add to calendar. update pointer to next progression
                nextevent = engine.Event('Progression',nextprog,subset_ProgressionPlayback[prog_index_pointer])
                Calendar.append(nextevent)
                prog_index_pointer = prog_index_pointer + 1
            else:
            #otherwise scheduling is done
                scheduling_done =1	
    

        # Simulation
        while Calendar != []:
            #obtains the next event of the calendar while also removing it from the calendar
            nextevent = Calendar.pop(0)	

            #if next event is year, call Year function to update simulation statistics
            if nextevent.type == 'Year':
                Sim.clock = nextevent.time
                print("A year has passed.")
                Sim, Stat = event.Year(Sim, Stat, reps)	

            #if next event is patient arrival, call arrival function to add patient to the waitlist of the DSA
            elif nextevent.type == 'Arrival':
                Sim.clock = nextevent.time
                Sim, Stat, OPTN = event.Arrival(nextevent.info, Sim, Stat, OPTN)	

            #if next event is progression, call progression function to update patient's status
            elif nextevent.type == 'Progression':
                Sim.clock = nextevent.time
                Sim, Stat, OPTN = event.Progression(nextevent.info, Sim, Stat, OPTN, reps)	

            #if next event is an organ arrival, call organ arrival function to allocate organ
            elif nextevent.type == 'Organ':
                Sim.clock = nextevent.time
                Sim, Stat, OPTN = event.OrganArrival(nextevent.info, Sim, Stat, OPTN, Regions, SharingPartners, Patients_Accept, Donor_Accept, DSA_Avg_Times, AcceptanceModelS1, AcceptanceModel, Relist, reps)	
    

        event.EndRep()	

        del Stat	
    

    #Output Results
    #convert arrays to data frames
    record_deaths = pd.DataFrame(data = Sim.record_deaths)
    record_deaths.columns = ['# of Deaths', 'Year', 'Replication #']	

    record_mr_disparity_mean = pd.DataFrame(data = Sim.record_mr_disparity_mean)
    record_mr_disparity_mean.columns = ['DSA Average Mortality Rate', 'Year', 'Replication #']	

    record_mr_disparity_std = pd.DataFrame(data = Sim.record_mr_disparity_std)
    record_mr_disparity_std.columns = ['DSA Mortality Rate Standard Deviation', 'Year', 'Replication #']	

    record_meld_disparity_mean = pd.DataFrame(data = Sim.record_meld_disparity_mean)
    record_meld_disparity_mean.columns = ['DSA Transplant Average MELD', 'Year', 'Replication #']	

    record_meld_disparity_std = pd.DataFrame(data = Sim.record_meld_disparity_std)
    record_meld_disparity_std.columns = ['Standard Deviation of Average DSA Transplant MELD', 'Year', 'Replication #']	

    record_medMELDmean = pd.DataFrame(data = Sim.record_medMELDmean)
    record_medMELDmean.columns = ['DSA Transplant MELD Median', 'Year', 'Replication #']	

    record_medMELDstd = pd.DataFrame(data = Sim.record_medMELDstd)
    record_medMELDstd.columns = ['Standard Deviation of DSA Transplant MELD Median', 'Year', 'Replication #']	

    DSA_column = ['Year', 'Replication #', 'Replication #']	

    for i in range(1, 59): DSA_column.append("DSA {0}".format(i))	

    record_ydeaths = pd.DataFrame(data = Sim.record_ydeaths)
    record_ydeaths.columns = DSA_column	

    record_ytransplants = pd.DataFrame(data = Sim.record_ytransplants)
    record_ytransplants.columns = DSA_column	

    record_yarrivals = pd.DataFrame(data = Sim.record_yarrivals)
    record_yarrivals.columns = DSA_column	

    record_ycandidates = pd.DataFrame(data = Sim.record_ycandidates)
    record_ycandidates.columns = DSA_column	

    record_yremoved = pd.DataFrame(data = Sim.record_yremoved)
    record_yremoved.columns = DSA_column	

    record_ywait = pd.DataFrame(data = Sim.record_ywait)
    record_ywait.columns = DSA_column	

    record_yMELD = pd.DataFrame(data = Sim.record_yMELD)
    record_yMELD.columns = DSA_column	

    record_txDSA = pd.DataFrame(data = Sim.record_txDSA)
    record_txDSAoutput = pd.DataFrame(data = Sim.record_txDSAoutput)	

    record_removals = pd.DataFrame(data = Sim.record_removals)
    record_removals.columns = ['Year', 'Replication #', 'Removal Time', 'Removed Patient ID', 'Patient Allocation MELD', 'Patient Lab MELD']	

    record_txID = pd.DataFrame(data = Sim.record_txID)
    record_txID.columns = ['Year', 'Replication #', 'Transplant Time', 'Transplant Patient ID', 'Regional Transplant', 'National Transplant']	

    record_doID = pd.DataFrame(data = Sim.record_doID)
    record_doID.columns = ['Year', 'Replication #', 'Transplant Time', 'Transplant Patient ID', 'Donor ID']	

    yrelists = pd.DataFrame(data = Sim.record_yrelists)
    yrelists.columns = DSA_column	

    yregrafts = pd.DataFrame(data = Sim.record_yregrafts)
    yregrafts.columns = DSA_column	

    record_txIDregraft = pd.DataFrame(data = Sim.record_txIDregraft)
    record_txIDregraft.columns = ['Year', 'Replication #', 'Re-Transplant Time', 'Re-Transplant Patient ID', 'Regional Re-Transplant', 'National Re-Transplant']	

    record_doIDregraft = pd.DataFrame(data = Sim.record_doIDregraft)
    record_doIDregraft.columns = ['Year', 'Replication #', 'Re-Transplant Time', 'Re-Transplant Patient ID', 'Donor ID']	

    record_relists = pd.DataFrame(data = Sim.record_relists)
    record_relists.columns = ['Year', 'Replication #', '1st Transplant Time', 'Patient ID', 'Patient Allocation MELD at First Transplant Time', 'Patient Earliest Re-Transplant Time']	

    #Output Results
    nump.savetxt(directory + "Output_deaths.txt", Sim.record_deaths, fmt='%1.4e', delimiter='\t', newline='\n')
    record_deaths.to_csv(directory + "Output_deaths.csv", sep=',', encoding='utf-8', index = False)	

    nump.savetxt(directory + "Output_mr_disparity_mean.txt", Sim.record_mr_disparity_mean, fmt='%1.4e', delimiter='\t', newline='\n')
    record_mr_disparity_mean.to_csv(directory + "Output_mr_disparity_mean.csv", sep=',', encoding='utf-8', index = False)	

    nump.savetxt(directory + "Output_mr_disparity_std.txt", Sim.record_mr_disparity_std, fmt='%1.4e', delimiter='\t', newline='\n')
    record_mr_disparity_std.to_csv(directory + "Output_mr_disparity_std.csv", sep=',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "Output_meld_disparity_mean.txt", Sim.record_meld_disparity_mean, fmt='%1.4e', delimiter='\t', newline='\n')
    record_meld_disparity_mean.to_csv(directory + "Output_meld_disparity_mean.csv", sep=',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "Output_meld_disparity_std.txt", Sim.record_meld_disparity_std, fmt='%1.4e', delimiter='\t', newline='\n')
    record_meld_disparity_std.to_csv(directory + "Output_meld_disparity_std.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "Output_meld_median_mean.txt", Sim.record_medMELDmean, fmt='%1.4e', delimiter='\t', newline='\n')
    record_medMELDmean.to_csv(directory + "Output_meld_median_mean.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "Output_meld_median_std.txt", Sim.record_medMELDstd, fmt='%1.4e', delimiter='\t', newline='\n')
    record_medMELDstd.to_csv(directory + "Output_meld_median_std.csv", sep =',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_ydeaths.txt", Sim.record_ydeaths, fmt='%1.4e', delimiter='\t', newline='\n')
    record_ydeaths.to_csv(directory + "RawOutput_ydeaths.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_ytransplants.txt", Sim.record_ytransplants, fmt='%1.4e', delimiter='\t', newline='\n')
    record_ytransplants.to_csv(directory + "RawOutput_ytransplants.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_yarrivals.txt", Sim.record_yarrivals, fmt='%1.4e', delimiter='\t', newline='\n')
    record_yarrivals.to_csv(directory + "RawOutput_yarrivals.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_ycandidates.txt", Sim.record_ycandidates, fmt='%1.4e', delimiter='\t', newline='\n')
    record_ycandidates.to_csv(directory + "RawOutput_ycandidates.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_yremoved.txt", Sim.record_yremoved, fmt='%1.4e', delimiter='\t', newline='\n')
    record_yremoved.to_csv(directory + "RawOutput_yremoved.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_ywait.txt", Sim.record_ywait, fmt='%1.4e', delimiter='\t', newline='\n')
    record_ywait.to_csv(directory + "RawOutput_ywait.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_yMELD.txt", Sim.record_yMELD, fmt='%1.4e', delimiter='\t', newline='\n')
    record_yMELD.to_csv(directory + "RawOutput_yMELD.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_DSAs.txt", Sim.record_txDSA, fmt='%1.4e', delimiter='\t', newline='\n')
    record_txDSA.to_csv(directory + "RawOutput_DSAs.csv", sep = ',', encoding = 'utf-8')	

    nump.savetxt(directory + "RawOutput_DSAs2.txt", Sim.record_txDSAoutput, fmt='%1.4e', delimiter='\t', newline='\n')
    record_txDSAoutput.to_csv(directory + "RawOutput_DSAs2.csv", sep = ',', encoding = 'utf-8')	

    nump.savetxt(directory + "RawOutput_removals.txt", Sim.record_removals, fmt='%1.4e', delimiter='\t', newline='\n')
    record_removals.to_csv(directory + "RawOutput_removals.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_TxID.txt", Sim.record_txID, fmt='%1.4e', delimiter='\t', newline='\n')
    record_txID.to_csv(directory + "RawOutput_TxID.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_DoID.txt", Sim.record_doID, fmt='%1.4e', delimiter='\t', newline='\n')
    record_doID.to_csv(directory + "RawOutput_DoID.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_yrelists.txt", Sim.record_yrelists, fmt='%1.4e', delimiter='\t', newline='\n')
    yrelists.to_csv(directory + "RawOutput_yrelists.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_yregrafts.txt", Sim.record_yregrafts, fmt='%1.4e', delimiter='\t', newline='\n')
    yregrafts.to_csv(directory + "RawOutput_yregrafts.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_TxIDregraft.txt", Sim.record_txIDregraft, fmt='%1.4e', delimiter='\t', newline='\n')
    record_txIDregraft.to_csv(directory + "RawOutput_TxIDregraft.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_DoIDregraft.txt", Sim.record_doIDregraft, fmt='%1.4e', delimiter='\t', newline='\n')
    record_doIDregraft.to_csv(directory + "RawOutput_DoIDregraft.csv", sep = ',', encoding = 'utf-8', index = False)	

    nump.savetxt(directory + "RawOutput_Relistid.txt", Sim.record_relists, fmt='%1.4e', delimiter='\t', newline='\n')
    record_relists.to_csv(directory + "RawOutput_Relistid.csv", sep = ',', encoding = 'utf-8', index = False)	

    print('Simulation Finished @ ',datetime.datetime.now().time())



if __name__ =="__main__":
    cli()
    