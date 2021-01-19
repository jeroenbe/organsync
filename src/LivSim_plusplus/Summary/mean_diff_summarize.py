# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:11:44 2017

@author: kbui1993
"""
import pandas as pd
import numpy as np
from scipy.stats import t

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

base_directory = "C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/base(cap_and_delay)/"
         
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

def compute_waitlist_death_diff(base_case, new_case):
    """
    This function computes the difference of deaths between the base case and
    another case. It appleis t-test to compute p-value.
    @Input:
        @base_case: base case death data set
        @new_case: new case deathd data set
    @Output:
        @diff: death difference
        @p_value: p value of the test
    """
    
    #count the number of observations in each case
    n_base = len(base_case)
    n_new = len(new_case)
    
    #compute the average number of deaths
    average_base = np.mean(base_case)
    average_new = np.mean(new_case)
    
    #compute the variance of deaths
    var_base = np.var(base_case)
    var_new = np.var(new_case)
    
    #compute the difference of deaths
    diff = average_new - average_base
    
    #compute the t score
    t_score =  np.absolute(diff)/np.sqrt(var_base/n_base+var_new/n_new)
    
    #compute degrees of freedom
    #df = ((var_base/n_base + var_new/n_new)**2)/(((var_base/n_base)**2)/(n_base-1) + ((var_new/n_new)**2)/(n_new-1))
    
    #compute p_value
    p_value = t.cdf(t_score, min(n_base-1, n_new-1))
    
    #return results
    return diff, 2*(1-p_value)
    

def compute_waitlist_removal_diff(base_case, new_case):
    """
    This function computes the difference of waitlist removal between the
    base case and another case. It applies t-test to compute p-value.
    @Input:
        @base_case: base case data set (should have 61 columns)
        @new_case: new case data set (should have 61 columns)
    @Output:
        @diff: waitlist removals difference
        @p_value: p value of the test
    """
    
    #count the number of observations in each case
    n_base = len(base_case)
    n_new = len(new_case)
    
    #obtain the row sum for each case
    row_sum_base = base_case.sum(axis = 1)
    row_sum_new = new_case.sum(axis = 1)
    
    #compute the average number of removals
    average_base = np.mean(row_sum_base)
    average_new = np.mean(row_sum_new)
    
    #compute the difference of deaths
    diff = average_new - average_base
    
    #compute the variance of removals
    var_base = np.var(row_sum_base)
    var_new = np.var(row_sum_new)
    
    #compute t-score
    t_score =  np.absolute(diff)/np.sqrt(var_base/n_base+var_new/n_new)
    
    #compute degrees of freedom
    #df = ((var_base/n_base + var_new/n_new)**2)/(((var_base/n_base)**2)/(n_base-1) + ((var_new/n_new)**2)/(n_new-1))
    
    #compute p-value
    p_value = t.cdf(t_score, min(n_base-1, n_new-1))
    
    #return result
    return diff, 2*(1-p_value)

def compute_diff_mean(base_case, new_case):
    """
    This function computes the mean difference and applies t-test to compute
    the p value.
    @Input:
        base_case: base case data set
        new_case: new case data set
    @Output:
        diff: mean difference
        p_value: p value of the t-test
    """
    
    #compute the number of observations for both cases
    n_base = len(base_case)
    n_new = len(new_case)
    
    #compute the average
    average_base = np.mean(base_case.iloc[:,0])
    average_new = np.mean(new_case.iloc[:,0])
    
    #compute the standard deviation
    var_base = np.var(base_case.iloc[:,0])
    var_new = np.var(new_case.iloc[:,0])
    
    #compute the difference of deaths
    diff = average_new - average_base
    
    #compute t-score
    t_score =  np.absolute(diff)/np.sqrt(var_base/n_base+var_new/n_new)
    
    #compute degrees of freedom
    #df = ((var_base/n_base + var_new/n_new)**2)/(((var_base/n_base)**2)/(n_base-1) + ((var_new/n_new)**2)/(n_new-1))
    
    #compute the p-value
    p_value = t.cdf(t_score, min(n_base-1, n_new-1))
    
    #return result
    return diff, 2*(1-p_value)

#read in total number of waitlist deaths for base case
death_base_case = pd.read_csv(base_directory + "Output_deaths.csv")
death_base_case = death_base_case.iloc[1:,0]

#read in waitlist removals for base case
waitlist_removal_base_case = pd.read_csv(base_directory + "RawOutput_yremoved.csv")
waitlist_removal_base_case = waitlist_removal_base_case.iloc[1:,3:]

#read in total number of post-transplant deaths for base case
posttx_death_base_case = pd.read_csv(base_directory + "Output_post_transplant_deaths.csv")
posttx_death_base_case = posttx_death_base_case.iloc[:,1]

#read in total number of retransplant deaths for base case
retx_death_base_case = pd.read_csv(base_directory + "Output_post_transplant_deaths_regrafts.csv")
retx_death_base_case = retx_death_base_case.iloc[:,1]

#read in total number of rewaitlist deaths for base case
rewaitlist_death_base_case = pd.read_csv(base_directory + "Output_waitlistrelist_deaths.csv")
rewaitlist_death_base_case = rewaitlist_death_base_case.iloc[:,1]

#read in mean meld for base case
mean_meld_base_data = pd.read_csv(base_directory + "Output_meld_disparity_mean.csv")
mean_meld_base_data = mean_meld_base_data.iloc[1:,]

#read in standard deviation of mean meld for base case
std_mean_meld_base_data = pd.read_csv(base_directory + "Output_meld_disparity_std.csv")
std_mean_meld_base_data = std_mean_meld_base_data.iloc[1:,]

#read in median meld for base case
median_meld_base_data = pd.read_csv(base_directory + "Output_meld_median_mean.csv")
median_meld_base_data = median_meld_base_data.iloc[1:,]

#read in standard deviation of median meld for base case
std_median_meld_base_data = pd.read_csv(base_directory + "Output_meld_median_std.csv")
std_median_meld_base_data = std_median_meld_base_data.iloc[1:,]

#read in average vehicle transport distance for base case
average_vehicle_transport_distance_base_data = pd.read_csv(base_directory + "AvgDistanceVehicle.csv")

#read in average helicopter transport distance for base case
average_helicopter_transport_distance_base_data = pd.read_csv(base_directory + "AvgDistanceHelicopter.csv")

#read in average airplane transport distance for base case
average_airplane_transport_distance_base_data = pd.read_csv(base_directory + "AvgDistanceAirplane.csv")

#read in average vehicle time for base case
average_vehicle_transport_time_base_data = pd.read_csv(base_directory + "AvgTimeVehicle.csv")

#read in average helicopter time for base case
average_helicopter_transport_time_base_data = pd.read_csv(base_directory + "AvgTimeHelicopter.csv")

#read in average airplane time for base case
average_airplane_transport_time_base_data = pd.read_csv(base_directory + "AvgTimeAirplane.csv")

#read in average percentage of organs transported by car for base case
average_car_percentage_base_data = pd.read_csv(base_directory + "CarPercentage.csv")

#read in average percentage of organs transported by helicopter for base case
average_helicopter_percentage_base_data = pd.read_csv(base_directory + "HelicopterPercentage.csv")

#read in average percentage of organs transported by airplane for base case
average_airplane_percentage_base_data = pd.read_csv(base_directory + "AirplanePercentage.csv")

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
death_diff_vector = []
death_pvalue_vector = []
waitlist_removal_mean_diff_vector = []
waitlist_removal_mean_diff_pvalue_vector = []
posttx_death_vector = []
posttx_death_pvalue = []
retx_death_vector = []
retx_death_pvalue = []
rewaitlist_death_vector = []
rewaitlist_death_pvalue = []
meld_mean_diff_vector = []
meld_mean_diff_pvalue_vector = []
std_meld_mean_diff_vector = []
std_meld_mean_diff_pvalue_vector = []
meld_median_diff_vector = []
meld_median_diff_pvalue_vector = []
std_median_meld_diff_vector = []
std_median_meld_pvalue_vector = []
avg_vehicle_transport_distance_vector = []
avg_vehicle_transport_distance_pvalue_vector = []
avg_vehicle_transport_time_vector = []
avg_vehicle_transport_time_pvalue_vector = []
avg_helicopter_transport_distance_vector = []
avg_helicopter_transport_distance_pvalue_vector = []
avg_helicopter_transport_time_vector = []
avg_helicopter_transport_time_pvalue_vector = []
avg_airplane_transport_distance_vector = []
avg_airplane_transport_distance_pvalue_vector = []
avg_airplane_transport_time_vector = []
avg_airplane_transport_time_pvalue_vector = []
avg_car_vector = []
avg_car_pvalue_vector = []
avg_helicopter_vector = []
avg_helicopter_pvalue_vector = []
avg_airplane_vector = []
avg_airplane_pvalue_vector = []

#begin computing mean differences
for i in range(0,len(files)):
    
    #compute the difference of waitlist death and p-value
    death_result = compute_waitlist_death_diff(death_base_case, num_of_waitlist_deaths[i])
    death_diff_vector.append(death_result[0])
    death_pvalue_vector.append(death_result[1])
    
    #compute the mean difference of waitlist removals and the p-value
    waitlist_removal_result = compute_waitlist_removal_diff(waitlist_removal_base_case, waitlist_removals[i])
    waitlist_removal_mean_diff_vector.append(waitlist_removal_result[0])
    waitlist_removal_mean_diff_pvalue_vector.append(waitlist_removal_result[1])

    #compute the mean difference of posttx death and p-value
    posttx_death_result = compute_waitlist_death_diff(posttx_death_base_case, num_of_posttx_deaths[i])
    posttx_death_vector.append(posttx_death_result[0])
    posttx_death_pvalue.append(posttx_death_result[1])

    #compute the mean difference of retransplant death and p-value
    retx_death_result = compute_waitlist_death_diff(retx_death_base_case, num_of_retx_deaths[i])
    retx_death_vector.append(retx_death_result[0])
    retx_death_pvalue.append(retx_death_result[1])

    #compute the mean difference of rewaitlist deaths and p-value
    rewaitlist_result = compute_waitlist_death_diff(rewaitlist_death_base_case, num_of_rewaitlist_deaths[i])
    rewaitlist_death_vector.append(rewaitlist_result[0])
    rewaitlist_death_pvalue.append(rewaitlist_result[1])

    #compute the mean difference of mean meld and the p-value
    meld_mean_diff_result = compute_diff_mean(mean_meld_base_data, mean_meld_data[i])
    meld_mean_diff_vector.append(meld_mean_diff_result[0])
    meld_mean_diff_pvalue_vector.append(meld_mean_diff_result[1])
    
    #compute the std of meld mean difference and p-value
    std_mean_meld_diff_result = compute_diff_mean(std_mean_meld_base_data, std_mean_meld_data[i])
    std_meld_mean_diff_vector.append(std_mean_meld_diff_result[0])
    std_meld_mean_diff_pvalue_vector.append(std_mean_meld_diff_result[1])
    
    #compute the mean difference of meld median and the p value
    meld_median_diff_result = compute_diff_mean(median_meld_base_data, median_meld_data[i])
    meld_median_diff_vector.append(meld_median_diff_result[0])
    meld_median_diff_pvalue_vector.append(meld_median_diff_result[1])

    #compute the standard deviation of meld median and the p value
    std_meld_median_diff_result = compute_diff_mean(std_median_meld_base_data, std_median_meld_data[i])
    std_median_meld_diff_vector.append(std_meld_median_diff_result[0])
    std_median_meld_pvalue_vector.append(std_meld_median_diff_result[1])

    #compute the mean difference of average vehicle transport distance
    avg_vehicle_transport_distance_result = compute_diff_mean(average_vehicle_transport_distance_base_data, avg_vehicle_transport_distance_data[i])
    avg_vehicle_transport_distance_vector.append(avg_vehicle_transport_distance_result[0])
    avg_vehicle_transport_distance_pvalue_vector.append(avg_vehicle_transport_distance_result[1])

    #compute the mean difference of average helicopter transport distance
    avg_helicopter_transport_distance_result = compute_diff_mean(average_helicopter_transport_distance_base_data, avg_helicopter_transport_distance_data[i])
    avg_helicopter_transport_distance_vector.append(avg_helicopter_transport_distance_result[0])
    avg_helicopter_transport_distance_pvalue_vector.append(avg_helicopter_transport_distance_result[1])

    #compute the mean difference of average airplane transport distance
    avg_airplane_transport_distance_result = compute_diff_mean(average_airplane_transport_distance_base_data, avg_airplane_transport_distance_data[i])
    avg_airplane_transport_distance_vector.append(avg_airplane_transport_distance_result[0])
    avg_airplane_transport_distance_pvalue_vector.append(avg_airplane_transport_distance_result[1])

    #compute the mean difference of average vehicle transport time
    avg_vehicle_transport_time_result = compute_diff_mean(average_vehicle_transport_time_base_data, avg_vehicle_transport_time_data[i])
    avg_vehicle_transport_time_vector.append(avg_vehicle_transport_time_result[0])
    avg_vehicle_transport_time_pvalue_vector.append(avg_vehicle_transport_time_result[1])

    #compute the mean difference of average helicopter transport time
    avg_helicopter_transport_time_result = compute_diff_mean(average_helicopter_transport_time_base_data, avg_helicopter_transport_time_data[i])
    avg_helicopter_transport_time_vector.append(avg_helicopter_transport_time_result[0])
    avg_helicopter_transport_time_pvalue_vector.append(avg_helicopter_transport_time_result[1])

    #compute the mean difference of average airplane transport time
    avg_airplane_transport_time_result = compute_diff_mean(average_airplane_transport_time_base_data, avg_airplane_transport_time_data[i])
    avg_airplane_transport_time_vector.append(avg_airplane_transport_time_result[0])
    avg_airplane_transport_time_pvalue_vector.append(avg_airplane_transport_time_result[1])

    #compute the mean difference of avg percentage of organs transported by car
    avg_car_result = compute_diff_mean(average_car_percentage_base_data, avg_car_percentage_data[i])
    avg_car_vector.append(avg_car_result[0])
    avg_car_pvalue_vector.append(avg_car_result[1])

    #compute the mean difference of avg percentage of organs transported by helicopters
    avg_helicopter_result = compute_diff_mean(average_helicopter_percentage_base_data, avg_helicopter_data[i])
    avg_helicopter_vector.append(avg_helicopter_result[0])
    avg_helicopter_pvalue_vector.append(avg_helicopter_result[1])

    #compute the mean difference of avg percentage of organs transported by airplanes
    avg_airplane_result = compute_diff_mean(average_airplane_percentage_base_data, avg_airplane_data[i])
    avg_airplane_vector.append(avg_airplane_result[0])
    avg_airplane_pvalue_vector.append(avg_airplane_result[1])


#preintialize summary data table
summary = []

#create summary table
summary.append(death_diff_vector)
summary.append(death_pvalue_vector)
summary.append(waitlist_removal_mean_diff_vector)
summary.append(waitlist_removal_mean_diff_pvalue_vector)
summary.append(posttx_death_vector)
summary.append(posttx_death_pvalue)
summary.append(retx_death_vector)
summary.append(retx_death_pvalue)
summary.append(rewaitlist_death_vector)
summary.append(rewaitlist_death_pvalue)
summary.append(meld_mean_diff_vector)
summary.append(meld_mean_diff_pvalue_vector)
summary.append(std_meld_mean_diff_vector)
summary.append(std_meld_mean_diff_pvalue_vector)
summary.append(meld_median_diff_vector)
summary.append(meld_median_diff_pvalue_vector)
summary.append(std_median_meld_diff_vector)
summary.append(std_median_meld_pvalue_vector)
summary.append(avg_vehicle_transport_distance_vector)
summary.append(avg_vehicle_transport_distance_pvalue_vector)
summary.append(avg_helicopter_transport_distance_vector)
summary.append(avg_helicopter_transport_distance_pvalue_vector)
summary.append(avg_airplane_transport_distance_vector)
summary.append(avg_airplane_transport_distance_pvalue_vector)
summary.append(avg_vehicle_transport_time_vector)
summary.append(avg_vehicle_transport_time_pvalue_vector)
summary.append(avg_helicopter_transport_time_vector)
summary.append(avg_helicopter_transport_time_pvalue_vector)
summary.append(avg_airplane_transport_time_vector)
summary.append(avg_airplane_transport_time_pvalue_vector)
summary.append(avg_car_vector)
summary.append(avg_car_pvalue_vector)
summary.append(avg_helicopter_vector)
summary.append(avg_helicopter_pvalue_vector)
summary.append(avg_airplane_vector)
summary.append(avg_airplane_pvalue_vector)

#convert to data frame
summary = pd.DataFrame(data = summary)

#name the columns
summary.columns = cases

#name the rows
rows = ['Annualized Waitlist Deaths', 'Annualized Waitlist Deaths p-value', 'Annualized Waitlist Removals',\
        'Annualized Waitlist Removals p-value', 'Annualized Post-Transplant Deaths', 'Annualized Post-Transplant Deaths p-value',\
        'Annualized ReTransplant Deaths', 'Annualized ReTransplant Deaths p-value', \
        'Annualized ReWaitlist Deaths', 'Annualized ReWaitlist Deaths p-value','DSA Mean Transplant MELD', \
        'DSA Mean Transplant MELD p-value', 'DSA Mean Transplant Standard Deviation',\
        'DSA Mean Transplant Standard Deviation p-value', 'DSA Median Transplant MELD',\
        'DSA Median Transplant MELD p-value', 'DSA Median Transplant MELD Standard Deviation',\
        'DSA Median Transplant MELD Standard Deviation p-value',\
        'Average Organ Vehicle Transport Distance', 'Average Organ Vehicle Transport Distance p-value',\
        'Average Organ Helicopter Transport Distance', 'Average Organ Helicopter Transport Distance p-value',\
        'Average Organ Airplane Transport Distance', 'Average Organ Airplane Transport Distance p-value',\
        'Average Organ Vehicle Transport Time', 'Average Organ Vehicle Transport Time p-value',\
        'Average Organ Helicopter Transport Time', 'Average Organ Helicopter Transport Time p-value',\
        'Average Organ Airplane Transport Time', 'Average Organ Airplane Transport Time p-value',\
        'Average Percentage Transported by Ground Vehicle', 'Average Percentage Transported by Ground Vehicle p-value',\
        'Average Percentage Transported by Helicopter', 'Average Percentage Transported by Helicopter p-value',\
        'Average Percentage Transported by Airplane', 'Average Percentage Transported by Airplane p-value']
summary.index = rows

summary.to_csv("C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/summary.csv")