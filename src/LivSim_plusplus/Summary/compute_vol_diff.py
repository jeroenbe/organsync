#import libraries
import csv
import pandas as pd
import numpy as np


#specify directory of default case
base_directory = "C:/Users/kbui1993/Desktop/Results - Copy/Liver Transplant (Boostingv2New)/base/"

#specify where to save files to
output_directory = "C:/Users/kbui1993/Desktop/Results - Copy/Liver Transplant (Boostingv2New)/"



#list of current DSA
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
cases = ['SRTR (Boostingv2New)',\
         'Share35 Share15 (Boostingv2New)',\
'Share35 Share18 (Boostingv2New)',\
'Share35 Share20 (Boostingv2New)',\
'Share37 Share15 (Boostingv2New)',\
'Share37 Share18 (Boostingv2New)',\
'Share37 Share20 (Boostingv2New)']

#list of directories corresponding to the cases
files = ['C:/Users/kbui1993/Desktop/Results - Copy/Liver Transplant (Boostingv2New)/SRTR/',\
             'C:/Users/kbui1993/Desktop/Results - Copy/Liver Transplant (Boostingv2New)/Share35_Share15_5boost/',\
             'C:/Users/kbui1993/Desktop/Results - Copy/Liver Transplant (Boostingv2New)/Share35_Share18_5boost/',\
             'C:/Users/kbui1993/Desktop/Results - Copy/Liver Transplant (Boostingv2New)/Share35_Share20_5boost/',\
             'C:/Users/kbui1993/Desktop/Results - Copy/Liver Transplant (Boostingv2New)/Share37_Share15_5boost/',\
             'C:/Users/kbui1993/Desktop/Results - Copy/Liver Transplant (Boostingv2New)/Share37_Share18_5boost/',\
             'C:/Users/kbui1993/Desktop/Results - Copy/Liver Transplant (Boostingv2New)/Share37_Share20_5boost/']

def compute_vol_diff(base_case, new_case):
    """
    This function computes the relative volume loss/gain for each DSA, i.e.
    (volume of new case - volume of base case)/(volume of base case) for each DSA.
    Input:
        @base_case: weighted matrix for the base case where entry a_ij is the 
        number of organs arriving from DSA i being transplanted at DSA j
        @new case: weighted matrix for the new case where entry a_ij is the number of 
        organs arriving from DSA i being transplanted at DSA j
    Output:
        @result: vector of relative volume loss/gain per DSA
    """
    
    #compute the sum along each columns for the base case and new case
    old_sum = base_case.sum(axis = 0)
    new_sum = new_case.sum(axis = 0)
    
    #compute relative volume gain/loss
    result = np.divide(np.subtract(new_sum, old_sum),old_sum)
    
    #convert to Data Frame
    result = pd.DataFrame(result)
    result = result.transpose()
    
    #name the columns
    result.columns = current_DSA
    
    #return result
    return result
    

def compute_percentile(new_case):
    """
    This function finds the minimum, 25th percentile, median,
    75th percentile, and maximum.
    Input:
        @new_case: result vector computed from compute_vol_diff
    Output:
        @result: vector containing the minimum, 25th percentile,..., and 
        maximum.
    """
    
    #preinitialize vector
    result = np.zeros((1,5))
    
    #find minimum, 25th percentile, ..., maximum
    result[0,0] = min(new_case.iloc[0,])
    result[0,1] = np.nanpercentile(new_case.iloc[0,], 25)
    result[0,2] = np.nanpercentile(new_case.iloc[0,], 50)
    result[0,3] = np.nanpercentile(new_case.iloc[0,], 75)
    result[0,4] = max(new_case.iloc[0,])
    
    #convert to Data Frame
    result = pd.DataFrame(data = result)
    
    #name the columns
    result.columns = ['minimum', '25th percentile', 'median',\
                      '75th percentile', 'maximum']
    
    #return result
    return result

#preinitialize list to contain DSA matrix for the base case
base_data = []

#open file containing the DSA matrix for the base case
with open(base_directory + 'RawOutput_DSAs.csv', encoding ='utf-8') as csvfile:
    data = csv.reader(csvfile, delimiter = ',')
    for row in data:
        base_data.append(row)

#convert entries to float
for i in range(1,59):
    base_data[i] = [float(j) for j in base_data[i]]

#convert to Data Frame and eliminate unnecessary row and column
base_data = pd.DataFrame(data = base_data)
base_data = base_data.iloc[1:,1:]

#preinitialize list to contain several DSA matrices for every cases
new_data = []

#iterate through list of files to store DSA matrix
for file in files:
    
    #preinitialize list to store DSA matrix
    input_data = []
    
    #open a file and record DSA matrix
    with open(file + "RawOutput_DSAs.csv") as csvfile:
        data = csv.reader(csvfile, delimiter = ',')
        for row in data:
            input_data.append(row)
        
    #convert entries of DSA matrix to float
    for i in range(1,59):
        input_data[i] = [float(j) for j in input_data[i]]
    
    #convert to Data Frame and eliminate unnecessary column and row
    input_data = pd.DataFrame(data = input_data)
    input_data = input_data.iloc[1:,1:]
    
    #add DSA matrix to list of DSA matrices
    new_data.append(input_data)

#preinitialize Data Frames for DSA volume gain/loss summary and general
#volume gain/loss summary
local_result = pd.DataFrame()
global_result = pd.DataFrame()

#iterate through list of DSA matrices 
for i in range(0, len(new_data)):
    #compute the relative volume gain/loss for each DSA given a DSA matrix
    DSA_vol = compute_vol_diff(base_data, new_data[i])
    
    #record DSA level specific volume gain/loss
    local_result = local_result.append(DSA_vol)
    
    #find the min, 25th percentile, ..., and maximum of the relative volume
    #gain/loss
    global_result = global_result.append(compute_percentile(DSA_vol))

#name the rows
local_result.index = cases
global_result.index = cases
    
#save as csv files
local_result.to_csv(output_directory + "DSA_Level_Transplant_Volume_Summary.csv")
global_result.to_csv(output_directory + "General_Transplant_Volume_Summary.csv")