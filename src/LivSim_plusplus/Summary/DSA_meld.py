# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 00:05:11 2017

@author: kbui1993
"""

import sys
import queue
import os
from copy import deepcopy
import pandas as pd
from matplotlib.dates import strpdate2num

#CHANGE DIRECTORIES HERE
base_directory = "C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/base(cap_and_delay)/"
output_directory = "C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/"

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

#list of cases
cases = ['default',\
         'SRTR',\
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
files = ['C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/base(cap_and_delay)/',\
         'C:/Users/kbui1993/Desktop/New Results/Cap_and_Delay/SRTR/',\
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


#create a list to store the total number of transplant per DSA across replication-years for each case
ytransplant = []

#create a list to store the total MELD per DSA across replication-years for each case
ymeld = []

for file in files:

   #read in transplant result
    ytransplant_case = pd.read_csv(file+"RawOutput_ytransplants.csv")

    #eliminate miscellaneous rows and columns
    ytransplant_case = ytransplant_case.iloc[1:,3:]

    #compute total transplant per DSA
    ytransplant_case = ytransplant_case.sum(axis = 0)

    #Add to the list
    ytransplant.append(ytransplant_case)
    

    #read in MELD result 
    ymeld_case = pd.read_csv(file+"RawOutput_yMELD.csv")
    
    #eliminate miscellaneous rows and columns
    ymeld_case = ymeld_case.iloc[1:,3:]

    #compute total MELD per DSA
    ymeld_case = ymeld_case.sum(axis = 0)

    #add to the list
    ymeld.append(ymeld_case)

#create a result data frame
result = []

for i in range(0, len(files)):

   #compute meld/transplant per DSA
    meld_vector = ymeld[i].div(ytransplant[i])

    #append vector to result
    result.append(meld_vector)

#convert to data frame
result = pd.DataFrame(result)

#name columns and rows
result.columns = current_DSA
result.index = cases

#write to csv file
result.to_csv(output_directory + "DSAmeld.csv")