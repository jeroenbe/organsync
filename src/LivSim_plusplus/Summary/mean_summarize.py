# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:11:44 2017

@author: kbui1993
"""
import pandas as pd
import numpy as np
from scipy.stats import t

#Change output directory here.
output_directory = "C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/"

#list of cases
cases = ['SRTR',\
         'Share29_Share15_0boost(8district)',\
         'Share29_Share18_3boost(8district)',\
         'Share29_Share20_5boost(8district)',\
         'Share35_Share15_0boost(8district)',\
         'Share35_Share15_3boost(8district)',\
         'Share35_Share15_5boost(8district)',\
         'Share35_Share18_3boost(8district)',\
         'Share35_Share20_5boost(8district)',\
         'Share29_Share15_0boost(11district)',\
         'Share29_Share18_3boost(11district)',\
         'Share29_Share20_5boost(11district)',\
         'Share35_Share18_3boost(11district)',\
         'Share35_Share20_5boost(11district)',\
         'Share35_Share18_0boost(11district)',\
         'Share35_Share20_0boost(11district)',\
         'Share29_Share15_0boost(400mile)',\
         'Share29_Share18_3boost(400mile)',\
         'Share29_Share20_5boost(400mile)',\
         'Share35_Share15_0boost(400mile)',\
         'Share35_Share15_3boost(400mile)',\
         'Share35_Share15_5boost(400mile)',\
         'Share35_Share18_0boost(400mile)',\
         'Share35_Share18_3boost(400mile)',\
         'Share35_Share18_5boost(400mile)',\
         'Share35_Share20_0boost(400mile)',\
         'Share35_Share20_3boost(400mile)',\
         'Share35_Share20_5boost(400mile)',\
         'Share35_Share22_0boost(400mile)',\
         'Share35_Share22_3boost(400mile)',\
         'Share35_Share22_5boost(400mile)',\
         'Share29_Share15_0boost(500mile)',\
         'Share29_Share18_3boost(500mile)',\
         'Share29_Share20_5boost(500mile)',\
         'Share35_Share15_0boost(500mile)',\
         'Share35_Share15_3boost(500mile)',\
         'Share35_Share15_5boost(500mile)',\
         'Share35_Share18_0boost(500mile)',\
         'Share35_Share18_3boost(500mile)',\
         'Share35_Share18_5boost(500mile)',\
         'Share35_Share20_0boost(500mile)',\
         'Share35_Share20_3boost(500mile)',\
         'Share35_Share20_5boost(500mile)',\
         'Share35_Share22_0boost(500mile)',\
         'Share35_Share22_3boost(500mile)',\
         'Share35_Share22_5boost(500mile)',\
         'Share29_Share15_0boost(600mile)',\
         'Share29_Share18_3boost(600mile)',\
         'Share29_Share20_5boost(600mile)',\
         'Share35_Share15_0boost(600mile)',\
         'Share35_Share15_3boost(600mile)',\
         'Share35_Share15_5boost(600mile)',\
         'Share35_Share18_0boost(600mile)',\
         'Share35_Share18_3boost(600mile)',\
         'Share35_Share18_5boost(600mile)',\
         'Share35_Share20_0boost(600mile)',\
         'Share35_Share20_3boost(600mile)',\
         'Share35_Share20_5boost(600mile)',\
         'Share35_Share22_0boost(600mile)',\
         'Share35_Share22_3boost(600mile)',\
         'Share35_Share22_5boost(600mile)',\
         'Share29_Share15_0boost(Constrained400mile)',\
         'Share29_Share18_3boost(Constrained400mile)',\
         'Share29_Share20_5boost(Constrained400mile)',\
         'Share35_Share15_0boost(Constrained400mile)',\
         'Share35_Share15_3boost(Constrained400mile)',\
         'Share35_Share15_5boost(Constrained400mile)',\
         'Share35_Share18_0boost(Constrained400mile)',\
         'Share35_Share18_3boost(Constrained400mile)',\
         'Share35_Share18_5boost(Constrained400mile)',\
         'Share35_Share20_0boost(Constrained400mile)',\
         'Share35_Share20_3boost(Constrained400mile)',\
         'Share35_Share20_5boost(Constrained400mile)',\
         'Share29_Share15_0boost(Constrained500mile)',\
         'Share29_Share18_3boost(Constrained500mile)',\
         'Share29_Share20_5boost(Constrained500mile)',\
         'Share35_Share15_0boost(Constrained500mile)',\
         'Share35_Share15_3boost(Constrained500mile)',\
         'Share35_Share15_5boost(Constrained500mile)',\
         'Share35_Share18_0boost(Constrained500mile)',\
         'Share35_Share18_3boost(Constrained500mile)',\
         'Share35_Share18_5boost(Constrained500mile)',\
         'Share35_Share20_0boost(Constrained500mile)',\
         'Share35_Share20_3boost(Constrained500mile)',\
         'Share35_Share20_5boost(Constrained500mile)',\
         'Share29_Share15_0boost(Constrained600mile)',\
         'Share29_Share18_3boost(Constrained600mile)',\
         'Share29_Share20_5boost(Constrained600mile)',\
         'Share35_Share15_0boost(Constrained600mile)',\
         'Share35_Share15_3boost(Constrained600mile)',\
         'Share35_Share15_5boost(Constrained600mile)',\
         'Share35_Share18_0boost(Constrained600mile)',\
         'Share35_Share18_3boost(Constrained600mile)',\
         'Share35_Share18_5boost(Constrained600mile)',\
         'Share35_Share20_0boost(Constrained600mile)',\
         'Share35_Share20_3boost(Constrained600mile)',\
         'Share35_Share20_5boost(Constrained600mile)']
         
#list of files
files = ['C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/SRTR/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/8district/Share29_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/8district/Share29_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/8district/Share29_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/8district/Share35_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/8district/Share35_Share15_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/8district/Share35_Share15_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/8district/Share35_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/8district/Share35_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/Current/Share29_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/Current/Share29_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/Current/Share29_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/Current/Share35_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/Current/Share35_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/Current/Share35_Share18_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/Current/Share35_Share20_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share29_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share29_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share29_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share15_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share15_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share18_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share18_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share20_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share20_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share22_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share22_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(400)/Share35_Share22_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share29_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share29_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share29_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share15_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share15_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share18_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share18_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share20_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share20_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share22_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share22_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(500)/Share35_Share22_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share29_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share29_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share29_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share15_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share15_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share18_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share18_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share20_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share20_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share22_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share22_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/LivSim(600)/Share35_Share22_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share29_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share29_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share29_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share35_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share35_Share15_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share35_Share15_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share35_Share18_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share35_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share35_Share18_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share35_Share20_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share35_Share20_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(400)/Share35_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share29_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share29_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share29_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share35_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share35_Share15_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share35_Share15_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share35_Share18_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share35_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share35_Share18_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share35_Share20_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share35_Share20_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(500)/Share35_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share29_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share29_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share29_Share20_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share35_Share15_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share35_Share15_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share35_Share15_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share35_Share18_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share35_Share18_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share35_Share18_5boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share35_Share20_0boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share35_Share20_3boost/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/ConstrainedLivSim(600)/Share35_Share20_5boost/']

def compute_mean_death(case):
    """
    This function computes the mean number of deaths:
    @Input:
        case: data of number of deaths per year
    @Output:
        @average_death: mean number of deaths
    """
    
    #compute the average number of deaths
    average_death = np.mean(case)
    
    #return results
    return average_death
    

def compute_mean_waitlist_removal(case):
    """
    This function computes the mean number of waitlist removal.
    @Input:
        @base_case: data set (should have 61 columns)
    @Output:
        @average_val: average number of waitlist removals
    """

    #obtain the row sum for each case
    row_sum_case = case.sum(axis = 1)
    
    #compute the average number of removals
    average_val = np.mean(row_sum_case)
    
    #return result
    return average_val

def compute_mean(case):
    """
    This function computes the mean from a given data set
    @Input:
        case: data of a case
    @Output:
		average_val: mean of the values of the given data
    """
    
    #compute the average
    average_case = np.mean(case.iloc[:,0])
    
    #return result
    return average_case

#preinitialize several lists to store data for other cases
num_of_waitlist_deaths = []
waitlist_removals = []
num_of_posttx_deaths = []
num_of_retx_deaths = []
num_of_rewaitlist_deaths = []
mean_meld_data = []
std_mean_meld_data = []
median_meld_data = []
std_median_meld_data = []
avg_vehicle_transport_distance_data = []
avg_helicopter_transport_distance_data = []
avg_airplane_transport_distance_data = []
avg_vehicle_transport_time_data = []
avg_helicopter_transport_time_data = []
avg_airplane_transport_time_data = []
avg_car_percentage_data = []
avg_helicopter_data = []
avg_airplane_data = []

#begin reading in other cases
for file in files:
    
    #read in number of waitlist deaths
    death_case_data = pd.read_csv(file+"Output_deaths.csv")
    death_case_data = death_case_data.iloc[1:,0]    
    num_of_waitlist_deaths.append(death_case_data)
    
    #read in waitlist removals
    waitlist_case_data = pd.read_csv(file+"RawOutput_yremoved.csv")
    waitlist_case_data = waitlist_case_data.iloc[1:,3:]
    waitlist_removals.append(waitlist_case_data)

    #read in total number of post-transplant deaths for base case
    posttx_death_case = pd.read_csv(file + "Output_post_transplant_deaths.csv")
    posttx_death_case = posttx_death_case.iloc[:,1]
    num_of_posttx_deaths.append(posttx_death_case)

    #read in total number of retransplant deaths for base case
    retx_death_case = pd.read_csv(file + "Output_post_transplant_deaths_regrafts.csv")
    retx_death_case = retx_death_case.iloc[:,1]
    num_of_retx_deaths.append(retx_death_case)

    #read in total number of rewaitlist deaths for base case
    rewaitlist_death_case = pd.read_csv(file + "Output_waitlistrelist_deaths.csv")
    rewaitlist_death_case = rewaitlist_death_case.iloc[:,1]
    num_of_rewaitlist_deaths.append(rewaitlist_death_case)

    #read in mean meld for a case
    mean_meld_case_data = pd.read_csv(file+"Output_meld_disparity_mean.csv")
    mean_meld_case_data = mean_meld_case_data.iloc[1:,]
    mean_meld_data.append(mean_meld_case_data)
    
    #read in standard deviation of mean meld for a case
    std_mean_meld_case = pd.read_csv(file+"Output_meld_disparity_std.csv")
    std_mean_meld_case= std_mean_meld_case.iloc[1:,]
    std_mean_meld_data.append(std_mean_meld_case)
    
    #read in median meld for a case
    median_meld_case_data = pd.read_csv(file+"Output_meld_median_mean.csv")
    median_meld_case_data = median_meld_case_data.iloc[1:,]
    median_meld_data.append(median_meld_case_data)
    
    #read in standard deviation of median meld for a case
    std_median_meld_case = pd.read_csv(file+"Output_meld_median_std.csv")
    std_median_meld_case = std_median_meld_case.iloc[1:,]
    std_median_meld_data.append(std_median_meld_case)

    #read in average vehicle transport distance data
    average_vehicle_transport_distance_case = pd.read_csv(file+"AvgDistanceVehicle.csv")
    avg_vehicle_transport_distance_data.append(average_vehicle_transport_distance_case)

    #read in average helicopter transport distance data
    average_helicopter_transport_distance_case = pd.read_csv(file+"AvgDistanceHelicopter.csv")
    avg_helicopter_transport_distance_data.append(average_helicopter_transport_distance_case)

    #read in average airplane transport distance data
    average_airplane_transport_distance_case = pd.read_csv(file+"AvgDistanceAirplane.csv")
    avg_airplane_transport_distance_data.append(average_airplane_transport_distance_case)

    #read in average vehicle transport time data
    average_vehicle_transport_time_case = pd.read_csv(file+"AvgTimeVehicle.csv")
    avg_vehicle_transport_time_data.append(average_vehicle_transport_time_case)

    #read in average helicopter transport time data
    average_helicopter_transport_time_case = pd.read_csv(file+"AvgTimeHelicopter.csv")
    avg_helicopter_transport_time_data.append(average_helicopter_transport_time_case)

    #read in average airplane transport time data
    average_airplane_transport_time_case = pd.read_csv(file+"AvgTimeAirplane.csv")
    avg_airplane_transport_time_data.append(average_airplane_transport_time_case)

    #read in average percentage of organs transported by car
    average_car_percentage_case = pd.read_csv(file+"CarPercentage.csv")
    avg_car_percentage_data.append(average_car_percentage_case)

    #read in average percentage of organs transported by helicopter
    average_helicopter_percentage_case = pd.read_csv(file+"HelicopterPercentage.csv")
    avg_helicopter_data.append(average_helicopter_percentage_case)

    #read in average percentage of organs transported by airplanes
    average_airplane_percentage_case = pd.read_csv(file+"AirplanePercentage.csv")
    avg_airplane_data.append(average_airplane_percentage_case)


    
#preinitialize a bunch of lists to store mean difference and p-values
waitlist_death_vector = []
waitlist_removal_mean_vector = []
posttx_death_vector = []
retx_death_vector = []
rewaitlist_death_vector = []
meld_mean_vector = []
std_meld_mean_vector = []
meld_median_vector = []
std_median_meld_vector = []
avg_vehicle_transport_distance_vector = []
avg_vehicle_transport_time_vector = []
avg_helicopter_transport_distance_vector = []
avg_helicopter_transport_time_vector = []
avg_airplane_transport_distance_vector = []
avg_airplane_transport_time_vector = []
avg_car_vector = []
avg_helicopter_vector = []
avg_airplane_vector = []

#begin computing mean differences
for i in range(0,len(files)):
    
    #compute the mean waitlist death
    waitlist_death_result = compute_mean_death(num_of_waitlist_deaths[i])
    waitlist_death_vector.append(waitlist_death_result)
    
    #compute the mean waitlist removals
    waitlist_removal_result = compute_mean_waitlist_removal(waitlist_removals[i])
    waitlist_removal_mean_vector.append(waitlist_removal_result)

    #compute the mean posttx death
    posttx_death_result = compute_mean_death(num_of_posttx_deaths[i])
    posttx_death_vector.append(posttx_death_result)

    #compute the mean retransplant death and p-value
    retx_death_result = compute_mean_death(num_of_retx_deaths[i])
    retx_death_vector.append(retx_death_result)

    #compute the mean difference of rewaitlist deaths and p-value
    rewaitlist_result = compute_mean_death(num_of_rewaitlist_deaths[i])
    rewaitlist_death_vector.append(rewaitlist_result)

    #compute the mean mean meld
    meld_mean_result = compute_mean(mean_meld_data[i])
    meld_mean_vector.append(meld_mean_result)
    
    #compute the mean std of meld mean
    std_mean_meld_result = compute_mean(std_mean_meld_data[i])
    std_meld_mean_vector.append(std_mean_meld_result)
    
    #compute the mean difference of meld median and the p value
    meld_median_result = compute_mean(median_meld_data[i])
    meld_median_vector.append(meld_median_result)

    #compute the standard deviation of meld median and the p value
    std_meld_median_result = compute_mean(std_median_meld_data[i])
    std_median_meld_vector.append(std_meld_median_result)

    #compute the mean difference of average vehicle transport distance
    avg_vehicle_transport_distance_result = compute_mean(avg_vehicle_transport_distance_data[i])
    avg_vehicle_transport_distance_vector.append(avg_vehicle_transport_distance_result)

    #compute the mean difference of average helicopter transport distance
    avg_helicopter_transport_distance_result = compute_mean(avg_helicopter_transport_distance_data[i])
    avg_helicopter_transport_distance_vector.append(avg_helicopter_transport_distance_result)

    #compute the mean difference of average airplane transport distance
    avg_airplane_transport_distance_result = compute_mean(avg_airplane_transport_distance_data[i])
    avg_airplane_transport_distance_vector.append(avg_airplane_transport_distance_result)

    #compute the mean difference of average vehicle transport time
    avg_vehicle_transport_time_result = compute_mean(avg_vehicle_transport_time_data[i])
    avg_vehicle_transport_time_vector.append(avg_vehicle_transport_time_result)

    #compute the mean difference of average helicopter transport time
    avg_helicopter_transport_time_result = compute_mean(avg_helicopter_transport_time_data[i])
    avg_helicopter_transport_time_vector.append(avg_helicopter_transport_time_result)

    #compute the mean difference of average airplane transport time
    avg_airplane_transport_time_result = compute_mean(avg_airplane_transport_time_data[i])
    avg_airplane_transport_time_vector.append(avg_airplane_transport_time_result)

    #compute the mean difference of avg percentage of organs transported by car
    avg_car_result = compute_mean(avg_car_percentage_data[i])
    avg_car_vector.append(avg_car_result)

    #compute the mean difference of avg percentage of organs transported by helicopters
    avg_helicopter_result = compute_mean(avg_helicopter_data[i])
    avg_helicopter_vector.append(avg_helicopter_result)

    #compute the mean difference of avg percentage of organs transported by airplanes
    avg_airplane_result = compute_mean(avg_airplane_data[i])
    avg_airplane_vector.append(avg_airplane_result)


#preintialize summary data table
summary = []

#create summary table
summary.append(waitlist_death_vector)
summary.append(waitlist_removal_mean_vector)
summary.append(posttx_death_vector)
summary.append(retx_death_vector)
summary.append(rewaitlist_death_vector)
summary.append(meld_mean_vector)
summary.append(std_meld_mean_vector)
summary.append(meld_median_vector)
summary.append(std_median_meld_vector)
summary.append(avg_vehicle_transport_distance_vector)
summary.append(avg_helicopter_transport_distance_vector)
summary.append(avg_airplane_transport_distance_vector)
summary.append(avg_vehicle_transport_time_vector)
summary.append(avg_helicopter_transport_time_vector)
summary.append(avg_airplane_transport_time_vector)
summary.append(avg_car_vector)
summary.append(avg_helicopter_vector)
summary.append(avg_airplane_vector)

#convert to data frame
summary = pd.DataFrame(data = summary)

#name the columns
summary.columns = cases

#name the rows
rows = ['Annualized Waitlist Deaths', 'Annualized Waitlist Removals',\
        'Annualized Post-Transplant Deaths', \
        'Annualized ReTransplant Deaths', \
        'Annualized ReWaitlist Deaths', 'DSA Mean Transplant MELD', \
        'DSA Mean Transplant Standard Deviation',\
        'DSA Median Transplant MELD',\
        'DSA Median Transplant MELD Standard Deviation',\
        'Average Organ Vehicle Transport Distance', \
        'Average Organ Helicopter Transport Distance', \
        'Average Organ Airplane Transport Distance', \
        'Average Organ Vehicle Transport Time', \
        'Average Organ Helicopter Transport Time', \
        'Average Organ Airplane Transport Time', \
        'Average Percentage Transported by Ground Vehicle', \
        'Average Percentage Transported by Helicopter', \
        'Average Percentage Transported by Airplane']
summary.index = rows

summary.to_csv(output_directory + "mean_summary2.csv")