import pandas as pd
import numpy as np

def estimate_post_transplant_outcome2(directory):
	"""
	This function estimates the number of deaths among patients who receive transplant.
	Results are written to file in the given directory.
	@Input:
		@directory: directory where the file RawOutput_DSAs2.csv is located and also where the output
		file will be written to.
	"""
	#upload average percentage of death per DSA
	avg_death_pct = pd.read_csv("C:/Users/kbui1993/Desktop/Postprocessing Input Files/prop.csv")

	#obtain the number of transplants across DSA for all 5 years of 5 replications
	transplants = pd.read_csv(directory+"RawOutput_DSAs2.csv")

	#get rid of the first 58 rows since they're all zeros
	transplants = transplants.iloc[58:,1:]

	#number of replications
	nreps = 5

	#initialize a list to store 5 replications of the total number of transplants in 5 years
	transplant_data_list = []

	#get the first replication of 5 years
	transplant_subset0 = transplants.iloc[232:289+1,:]
	transplant_data_list.append(np.sum(transplant_subset0, axis = 0))

	#get total number of transplants at the end of 5 years per replication
	for n in range(1,nreps):
		transplant_subset1 = transplants.iloc[(232+(58*5)*n):(289+(58*5)*n)+1,:]
		total_transplant_subset = np.subtract(transplant_subset1, transplant_subset0)
		transplant_data_list.append(np.sum(total_transplant_subset, axis = 0))
		transplant_subset0 = transplant_subset1

	#intialize list to store total number of deaths per 5 years
	result= []
	avg_dsa_death = avg_death_pct.ix[:,1]

	#compute total number of deaths per 5 years for each replication
	for n in range(0,nreps):
		result.append(np.sum(np.multiply(avg_dsa_death, transplant_data_list[n])))

	#divide by 5 to get average post-tx death per year
	result = np.divide(result,5)

	#convert to data frame
	result = pd.DataFrame(result)

	#name the column
	result.columns = ["Number of Deaths"]

	#write to csv file
	result.to_csv(directory + 'Output_post_transplant_deaths2.csv')
