import numpy as nump
import time
import csv
import scipy as scip
import datetime
import operator
import sys
import queue
import os
from copy import deepcopy
import pandas as pd
from matplotlib.dates import strpdate2num

current_DSA = ['ALOB', 'AROR', 'AZOB', 'CADN', 'CAGS', \
			   'CAOP', 'CASD', 'CORS', 'CTOP', 'DCTC',\
			   'FLFH', 'FLMP', 'FLUF', 'FLWC', 'GALL', \
			   'HIOP', 'IAOP', 'ILIP', 'INOP', 'KYDA', \
			   'LAOP', 'MAOB', 'MDPC', 'MIOP', 'MNOP', \
			   'MOMA', 'MSOP', 'MWOB', 'NCCM', 'NCNC', \
			   'NEOR', 'NJTO', 'NMOP', 'NVLV', 'NYAP', \
			   'NYFL', 'NYRT', 'NYWN', 'OHLB', 'OHLC', \
			   'OHLP', 'OHOV', 'OKOP', 'ORUO', 'PADV', \
			   'PATF', 'PRLL', 'SCOP', 'TNDS', 'TNMS', \
			   'TXGC', 'TXSA', 'TXSB', 'UTOP', 'VATB', \
			   'WALC', 'WIDN', 'WIUW']

def estimate_transport(transplants):
    """
    This function computes transport statistics for organs traveling between DSAs
    @Input:
        @transplants: data frame containing number of organs being transported 
        from DSA i to DSA j across 5 years 5 replications
    @Output:
        @distances: average distance traveled by an organ
        @times: average time traveled by an organ
        @drives: percentage of organ transported by ground vehicle
        @helicopters: percentage of organ transported by helicopters
        @airplanes: percentage of organ transported by airplanes
    """

    #Settings
    nreps = 5
    nump.random.seed(7777)
    ndsa = 58

    #load distance-time-mode data
    data = nump.loadtxt("distancetimes.txt")

    #Setup data
    dis_data = [[[] for i in range(0,ndsa)] for j in range(0,ndsa)]
    time_data = [[[] for i in range(0,ndsa)] for j in range(0,ndsa)]
    mode_data = [[[] for i in range(0,ndsa)] for j in range(0,ndsa)]
    
    total_transplant = transplants.iloc[(232+(58*5)*4):(289+(58*5)*4)+1,:]
    total_transplant = total_transplant.sum(axis = 1)

    for i in range(0,nump.shape(data)[0]):
        opo = int(data[i,0])
        txdsa =  int(data[i,1])
        dis_data[opo][txdsa].append(data[i,2])
        time_data[opo][txdsa].append(data[i,3])
        mode_data[opo][txdsa].append(data[i,4])

    #Prepare Output
    distances_vehicle = []
    distances_helicopter = []
    distances_airplane = []
    times_vehicle = []
    times_helicopter = []
    times_airplane = []    
    drives =[]
    helicopters = []
    airplanes = []

    dsa_distances_vehicle = nump.zeros((ndsa))
    dsa_distances_helicopter = nump.zeros((ndsa))
    dsa_distances_airplane = nump.zeros((ndsa))

    dsa_times_vehicle = nump.zeros((ndsa))
    dsa_times_helicopter = nump.zeros((ndsa))
    dsa_times_airplane = nump.zeros((ndsa))


    #preinitialize list to store number of transplant at the end of 5 years per replication
    transplant_data_list = []
    transplant_subset0 = transplants.iloc[232:289+1,:]
    transplant_data_list.append(transplant_subset0)


    #get total number of transplants at the end of 5 years per replication
    for n in range(1,nreps):
        transplant_subset1 = transplants.iloc[(232+(58*5)*n):(289+(58*5)*n)+1,:]
        total_transplant_subset = nump.subtract(transplant_subset1, transplant_subset0)
        transplant_data_list.append(total_transplant_subset)
        transplant_subset0 = transplant_subset1

    for n in range(0,nreps):

        #get tx data of replication n
        tx_subset = transplant_data_list[n]

        #get total number of transplants across 5 years in one replication
        tx_total = nump.sum(tx_subset.sum())

        #pre-initialize variables to store relevant values
        moment1_distance_vehicle = 0
        moment1_distance_helicopter = 0
        moment1_distance_airplane = 0
        moment1_time_vehicle = 0
        moment1_time_helicopter = 0
        moment1_time_airplane = 0

        count_drive = 0
        count_helicopter = 0
        count_airplane = 0

        #iterate through each entry of DSA i to DSA j to estimate distance traveled and hours traveled for
        #an organ
        for i in range(0,ndsa):
            for j in range(0,ndsa):
                #skip if there is negative number of transplants (which isn't supposed to happen)
                if int(tx_subset.iloc[i,j]) <= 0:
                    pass
                else:
                    for k in range(0,int(tx_subset.iloc[i,j])):
                        if len(dis_data[i][j]) > 0:
                            #select DSA - donor hospital pair randomly
                            randindex = nump.random.choice(list(range(0,len(dis_data[i][j]))))
                            
                            #update stats
                            if(mode_data[i][j][randindex] == 0):
                                count_drive = count_drive + 1
                                moment1_distance_vehicle = moment1_distance_vehicle + dis_data[i][j][randindex]
                                moment1_time_vehicle = moment1_time_vehicle + time_data[i][j][randindex]
                                dsa_distances_vehicle[i] = dsa_distances_vehicle[i] + dis_data[i][j][randindex]
                                dsa_times_vehicle[i] = dsa_times_vehicle[i] + time_data[i][j][randindex]
                            elif(mode_data[i][j][randindex] == 1):
                                count_helicopter = count_helicopter + 1
                                moment1_distance_helicopter = moment1_distance_helicopter + dis_data[i][j][randindex]
                                moment1_time_helicopter = moment1_time_helicopter + time_data[i][j][randindex]
                                dsa_distances_helicopter[i] = dsa_distances_helicopter[i] + dis_data[i][j][randindex]
                                dsa_times_helicopter[i] = dsa_times_helicopter[i] + time_data[i][j][randindex]
                            elif(mode_data[i][j][randindex] == 2):
                                count_airplane = count_airplane + 1
                                moment1_distance_airplane = moment1_distance_airplane + dis_data[i][j][randindex]
                                moment1_time_airplane = moment1_time_airplane + time_data[i][j][randindex]
                                dsa_distances_airplane[i] = dsa_distances_airplane[i] + dis_data[i][j][randindex]
                                dsa_times_airplane[i] = dsa_times_airplane[i] + time_data[i][j][randindex]

        #store average statistics
        distances_vehicle.append(moment1_distance_vehicle/count_drive)
        distances_helicopter.append(moment1_distance_helicopter/count_helicopter)
        distances_airplane.append(moment1_distance_airplane/count_airplane)

        times_vehicle.append(moment1_time_vehicle/count_drive)
        times_helicopter.append(moment1_time_helicopter/count_helicopter)
        times_airplane.append(moment1_time_airplane/count_airplane)

        drives.append(count_drive/tx_total)
        helicopters.append(count_helicopter/tx_total)
        airplanes.append(count_airplane/tx_total)
    
    

    #convert to data frames
    distances_vehicle = pd.DataFrame(distances_vehicle)
    distances_vehicle.columns = ['Average Distance Traveled on Ground']
    distances_helicopter = pd.DataFrame(distances_helicopter)
    distances_helicopter.columns = ['Average Distance Traveled by Helicopter']
    distances_airplane = pd.DataFrame(distances_airplane)
    distances_airplane.columns = ['Average Distance Traveled by Airplane']
    dsa_distances_vehicle = nump.divide(dsa_distances_vehicle, total_transplant)
    dsa_distances_vehicle = pd.DataFrame(dsa_distances_vehicle)
    dsa_distances_vehicle.columns = ['Average Distance Traveled on Ground']
    dsa_distances_vehicle.index = current_DSA
    dsa_distances_helicopter = nump.divide(dsa_distances_helicopter, total_transplant)
    dsa_distances_helicopter = pd.DataFrame(dsa_distances_helicopter)
    dsa_distances_helicopter.columns = ['Average Distance Traveled by Helicopter']
    dsa_distances_helicopter.index = current_DSA
    dsa_distances_airplane = nump.divide(dsa_distances_airplane, total_transplant)
    dsa_distances_airplane = pd.DataFrame(dsa_distances_airplane)
    dsa_distances_airplane.columns = ['Average Distance Traveled by Airplane']
    dsa_distances_airplane.index = current_DSA

    times_vehicle = pd.DataFrame(times_vehicle)
    times_vehicle.columns = ['Average Time Traveled on Ground']
    times_helicopter = pd.DataFrame(times_helicopter)
    times_helicopter.columns = ['Average Time Traveled by Helicopter']
    times_airplane = pd.DataFrame(times_airplane)
    times_airplane.columns = ['Average Time Traveled by Airplane']
    dsa_times_vehicle = nump.divide(dsa_times_vehicle, total_transplant)
    dsa_times_vehicle = pd.DataFrame(dsa_times_vehicle)
    dsa_times_vehicle.columns = ['Average Time Traveled on Ground']
    dsa_times_vehicle.index = current_DSA
    dsa_times_helicopter = nump.divide(dsa_times_helicopter, total_transplant)
    dsa_times_helicopter = pd.DataFrame(dsa_times_helicopter)
    dsa_times_helicopter.columns = ['Average Time Traveled by Helicopter']
    dsa_times_helicopter.index = current_DSA
    dsa_times_airplane = nump.divide(dsa_times_airplane, total_transplant)
    dsa_times_airplane = pd.DataFrame(dsa_times_airplane)
    dsa_times_airplane.columns = ['Average Time Traveled by Airplane']
    dsa_times_airplane.index = current_DSA
    

    drives = pd.DataFrame(drives)
    drives.columns = ['Percentage of Organ Traveled by Car']
    helicopters = pd.DataFrame(helicopters)
    helicopters.columns = ['Percentage of Organ Traveled by Helicopter']
    airplanes = pd.DataFrame(airplanes)
    airplanes.columns = ['Percentage of Organ Traveled by Airplane']

    #return results
    return distances_vehicle, distances_helicopter, distances_airplane, times_vehicle, times_helicopter, times_airplane, drives, helicopters, airplanes, dsa_distances_vehicle, dsa_distances_helicopter, dsa_distances_airplane, dsa_times_vehicle, dsa_times_helicopter, dsa_times_airplane


def output_distance_data(directory):
    """
    This function estimates the transport statistics based on RawOutput_DSAs2.csv. 
    It writes the results into the directory
    @Input:
        @directory: location of RawOutput_DSAs2.csv. Also location where
        output files will be written to.
    """

    #read in transplants data
    transplants = pd.read_csv(directory + "RawOutput_DSAs2.csv")
    transplants = transplants.iloc[58:, 1:]

    #compute travel statistics
    travel_stat = estimate_transport(transplants)

    #write results to directory
    travel_stat[0].to_csv(directory+"AvgDistanceVehicle.csv", sep = ',', encoding = 'utf-8', index = False)
    travel_stat[1].to_csv(directory+"AvgDistanceHelicopter.csv", sep = ',', encoding = 'utf-8', index = False)
    travel_stat[2].to_csv(directory+"AvgDistanceAirplane.csv", sep = ',', encoding = 'utf-8', index = False)
    travel_stat[3].to_csv(directory+"AvgTimeVehicle.csv", sep = ',', encoding = 'utf-8', index = False)
    travel_stat[4].to_csv(directory+"AvgTimeHelicopter.csv", sep = ',', encoding = 'utf-8', index = False)
    travel_stat[5].to_csv(directory+"AvgTimeAirplane.csv", sep = ',', encoding = 'utf-8', index = False)
    travel_stat[6].to_csv(directory+"CarPercentage.csv", sep = ',', encoding = 'utf-8', index = False)
    travel_stat[7].to_csv(directory+"HelicopterPercentage.csv", sep = ',', encoding = 'utf-8', index = False)
    travel_stat[8].to_csv(directory+"AirplanePercentage.csv", sep = ',', encoding = 'utf-8', index = False)
    travel_stat[9].to_csv(directory+"DSAAvgDistanceVehicle.csv", sep =',', encoding = 'utf-8')
    travel_stat[10].to_csv(directory+"DSAAvgDistanceHelicopter.csv", sep =',', encoding = 'utf-8')
    travel_stat[11].to_csv(directory+"DSAAvgDistanceAirplane.csv", sep =',', encoding = 'utf-8')
    travel_stat[12].to_csv(directory+"DSAAvgTimeVehicle.csv", sep =',', encoding = 'utf-8')
    travel_stat[13].to_csv(directory+"DSAAvgTimeHelicopter.csv", sep =',', encoding = 'utf-8')
    travel_stat[14].to_csv(directory+"DSAAvgTimeAirplane.csv", sep =',', encoding = 'utf-8')











