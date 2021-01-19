#This code prepares and estimates post-transplant deaths
import numpy as nump
import time
import csv
import scipy as scip
import datetime
import operator
import sys
import queue
import pandas as pd
from copy import deepcopy
from matplotlib.dates import strpdate2num

#Prepare Converters
def tf_convert(s):
    #print(s)
    if s ==b'True':
        return 1
    else:
        return 0

def missing_convert(s):
    #print(s)
    if s ==b'.':
        return nump.nan
    else:
        return float(s)


def estimate_post_transplant_death(txids, doids):
	"""
	This function estimate the number of post transplant deaths.
	@Input:
		@txids: list of patients who received transplants
		@doids: list of donated organs
	@Output:
		@output_totals: number of post transplant deaths for each replication
	"""

	#set seed
	nump.random.seed(7777)

	patcols = (2,5,11,71,72,74,79,82,86,88,95,97,104,106,107,108,109,110,135,138) #patient columns
	statuscols = (1,4,5,8,9,14,15) #status columns
	istatuscols = {2,120,121,124,125,134,98} #
	donorcols =(4,15,54,55,57,60,70,76,82,85) #donor columns

	waitlist = nump.loadtxt("waitlist.txt",delimiter ="|",skiprows=3,usecols=patcols,converters={11: tf_convert}) #upload waitlist file
	patients = nump.loadtxt("patients.txt",delimiter ="|",skiprows=3,usecols=patcols,converters={11: tf_convert}) #upload patient file
	patients = nump.vstack((waitlist,patients)) #concatenate patient to waitlist

	#do the same as before
	is_waitlist = nump.loadtxt("waitlist.txt",delimiter ="|",skiprows=3,usecols=istatuscols,converters={120: missing_convert,121: missing_convert,124: missing_convert,125: missing_convert, 134: missing_convert,98: missing_convert })
	is_patients = nump.loadtxt("patients.txt",delimiter ="|",skiprows=3,usecols=istatuscols,converters={120: missing_convert,121: missing_convert,124: missing_convert,125: missing_convert, 134: missing_convert,98: missing_convert })
	is_patients = nump.vstack((is_waitlist,is_patients))

	#upload status and status time
	status= nump.loadtxt("status.txt",delimiter ="|",skiprows=3,usecols=statuscols,converters={4: missing_convert,5: missing_convert,8: missing_convert,9: missing_convert,14: missing_convert,15: missing_convert })
	statustimes = nump.loadtxt("status_times.txt")

	#upload donors
	donors = nump.loadtxt("donor.txt",delimiter ="|",skiprows=3,usecols=donorcols)

	#survival coefficients and step survival function
	survcoeff = nump.loadtxt("survivalcoefficients.txt")
	stepsurv = nump.loadtxt("stepsurvival.txt")

	#Setting
	nreps = 1 # number of replications
	maxtime = 5 #maxtime of survival
	output_totals = []

	for i in range(0,nreps):
		for y in range(0, maxtime):
			#print iteration
			print('Replication %d, Year %d' %(i,y))
		
			#Form Survival Dataset
			survdata = nump.empty([1,50])
			txtimes = nump.empty([1,1])

			#get donors of replication i and by year y
			donor_subset = doids[doids.iloc[:,1] == i+1]
			donor_subset = donor_subset[donor_subset.iloc[:,0] == y]

			#get transplant patient of replication i
			tx_subset = txids[txids.iloc[:,1] == i+1]
			tx_subset = tx_subset[tx_subset.iloc[:,0] == y]

			for n in range(0, len(donor_subset)):
				lsampatid = int(donor_subset.iloc[n,3]) #lsam patient id
				lsamdonid = int(donor_subset.iloc[n,4]) -1 #lsam donor id
				lsamtxtime = donor_subset.iloc[n,2] #lsam transplant time

				#Create categorical age variables
				page = [1*(patients[lsampatid][1] < 18), 1*((patients[lsampatid][1] >= 18 and patients[lsampatid][1] <25)),
				1*((patients[lsampatid][1] >= 25 and patients[lsampatid][1] <35)),
				1*((patients[lsampatid][1] >= 45 and patients[lsampatid][1] <55)),
				1*((patients[lsampatid][1] >= 55 and patients[lsampatid][1] <65)),
				1*((patients[lsampatid][1] >= 65))] #patient age

				dage = [1*(donors[lsamdonid][0] < 18), 1*((donors[lsamdonid][0] >= 40 and donors[lsamdonid][0] <50)),
				1*((donors[lsamdonid][0] >= 50 and donors[lsamdonid][0] <60)),
				1*((donors[lsamdonid][0] >= 60 and donors[lsamdonid][0] <70)),
				1*((donors[lsamdonid][0] >= 70))] #donor age

				#Obtain last status record before transplant
				statuspat = is_patients[lsampatid]
				for j in range(1,len(statustimes)):
					if statustimes[j][1] > lsamtxtime:
						"""
						skip the patient whose status time is later than transplant time
						"""
						break

					if statustimes[j][0] == lsampatid and nump.isnan(status[j]).any() == False:
						statuspat = status[j]

				record = nump.hstack((patients[lsampatid],donors[lsamdonid],statuspat,tx_subset.iloc[n,4],tx_subset.iloc[n,5],page,dage))
				survdata = nump.vstack((survdata,record)) #append the observation to the survival data
				txtimes = nump.vstack((txtimes,lsamtxtime)) #append the transplant time

			#get rid of the first row that is just zeroes
			survdata = survdata[1:]
			txtimes = txtimes[1:]

			#Compute survival
			values = nump.zeros(nump.shape(survdata)[0])
			for l in range(0, nump.shape(survdata)[0]):
				values[l] = nump.exp(nump.dot(survdata[l],survcoeff))

			mobs = nump.shape(survdata)[0] #obtain number of observations

			svalues = deepcopy(values)
			deaths = deepcopy(values)

			mu = nump.random.uniform(0,1,mobs) #create a vector of probability values from [0,1]

			for m in range(0,mobs):
				svalues[m] = nump.exp(nump.log(mu[m])/values[m])

				#Calculate death
				for k in range(1,nump.shape(stepsurv)[0]):
					if svalues[m] < stepsurv[-1,0]:
						svalues[m] = stepsurv[-1,1]
						#deaths[m] = int(bool( nump.random.uniform(0,1,1) <=stepsurv[-1,2]  and svalues[m]/365 + txtimes[m]  <=maxtime))
						deaths[m] = int(bool(  svalues[m]/365 + txtimes[m]  <=maxtime))
						break
					elif svalues[m] < stepsurv[k-1,0] and svalues[m] >= stepsurv[k,0]:
						svalues[m] = stepsurv[k,1]
						#deaths[m] = int(bool( nump.random.uniform(0,1,1) <=stepsurv[k,2]  and svalues[m]/365 + txtimes[m]  <=maxtime))
						deaths[m] = int(bool(  svalues[m]/365 + txtimes[m] <=maxtime))
						break

			#Total Deaths
			output_totals.append(nump.sum(deaths))

	return output_totals

def estimate_post_transplant_outcome(directory):
	"""
	This function estimates the number of deaths among patients who receive transplant.
	Results are writtent to file in the given directory.
	@Input:
		@directory: directory where the files RawOutput_TxID.csv and RawOutput_DoID.csv are located. Also where the output
		files will be written to.
	"""

	txids = pd.read_csv(directory + 'RawOutput_TxID.csv')
	doids = pd.read_csv(directory + 'RawOutput_DoID.csv')

	#estimate post transplant deaths
	total_transplant_death = estimate_post_transplant_death(txids, doids)

	#write to csv file
	total_transplant_death = pd.DataFrame(total_transplant_death)
	total_transplant_death.columns = ["Number of Transplant Death"]
	total_transplant_death.to_csv(directory + 'Output_post_transplant_deaths.csv')









