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
    #True is coded as 1
    if s ==b'True':
        return 1
    #False is coded as 0
    else:
        return 0

def missing_convert(s):
    #print(s)
    #. is coded as NaN
    if s ==b'.':
        return nump.nan
    #any number is converted to a float
    else:
        return float(s)

def estimate_waitlist_relist_death(relist, regraft):
	""""
	This function estimates the number of deaths due to waitlist relist
	@Input:
		@relist: list of patients who are relisted for transplant
		@regraft: list of patients who received retransplant
	@Output:
		@total_relistdeaths: number of waitlisted patients who died before obtaining re-transplant
	"""
	#set seed
	nump.random.seed(7777)

	#setting
	maxtime = 5 #maxtime
	nreps = 5 #number of replications
	dprob = 0.152067281 #probability of death
	
	#subset the relist data set for patient ids and earliest re-transplant time
	relistids = relist.iloc[:,(0,1,3,5)]

	#subset the regraft for the patient id
	regraftids = regraft.iloc[:,3]
	regraftids = pd.DataFrame(regraftids)

	#cross reference the patient id in the relist data set of patient ids and the regraft data set of patient ids
	relistids_nografts_uncensored = relistids[~relistids.iloc[:,2].isin(regraftids.iloc[:,0])]
	relistids_nografts_uncensored = relistids_nografts_uncensored[relistids_nografts_uncensored.iloc[:,3] < maxtime]

	#preinitialize list of death counts for each replication
	total_relistdeaths = []

	for n in range(0,nreps):
		for y in range(0,maxtime):
			#extract the subset of people of replication n+1
			subset = relistids_nografts_uncensored[relistids_nografts_uncensored.iloc[:,1] == n+1]
			subset = subset[subset.iloc[:,0] == y]

			#create a vector of random numberrs between 0 and 1 length number of people relisted but not transplanted per replication
			r1 = nump.random.uniform(0,1, len(subset))

			#count the number of people who died
			count = sum([int(x < dprob) for x in r1])

			#store the number of deaths per replication
			total_relistdeaths.append(count)

	#return the results
	return total_relistdeaths


def estimate_post_retransplant_death(txids, doids):
	"""
	This function estimate the number of post retransplant deaths.
	@Input:
		@txids: list of patients who received retransplant
		@doids: list of donated organs
	@Output:
		@output_totals: number of retransplant deaths for each replication
	"""
	#set seed
	nump.random.seed(7777)

	patcols = (2,5,11,71,72,74,79,82,86,88,95,97,104,106,107,108,109,110,135,138) #patient columns
	statuscols = (1,4,5,8,9,14,15) #status columns
	istatuscols = {2,120,121,124,125,134,98} #
	donorcols =(4,15,54,55,57,60,70,76,82,85) #donor columns

	waitlist = nump.loadtxt("waitlist.txt",delimiter ="|",skiprows=3,usecols=patcols,converters={11: tf_convert}) #upload waitlist file
	patients = nump.loadtxt("patients.txt",delimiter ="|",skiprows=3,usecols=patcols,converters={11: tf_convert}) #upload patient file
	patients = nump.vstack((waitlist,patients)) #concatenate patient to waitlist file

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
	nreps = 5 #number of replications
	maxtime = 5 #maxtime of survival
	output_totals = [] #preinitialize list to record number of deaths

	for i in range(0,nreps):
		for y in range(0,maxtime):
			#Form survival dataset
			survdata = nump.empty([1,50])
			txtimes = nump.empty([1,1])

			#get donors of replication i
			donor_subset = doids[doids.iloc[:,1] == i+1]
			donor_subset = donor_subset[donor_subset.iloc[:,0] == y]

			#get transplant patient of replication i
			tx_subset = txids[txids.iloc[:,1] == i+1]
			tx_subset = tx_subset[tx_subset.iloc[:,0] == y]
		
			for n in range(0, len(donor_subset)):
				lsampatid = int(donor_subset.iloc[n,3]) #lsam patient id
				lsamdonid = int(donor_subset.iloc[n,4]) -1 #lsam donor id
				lsamtxtime = donor_subset.iloc[n,2] #lsam transplant time

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
						"""skip patient whose status time is later than the transplant time
						"""
						break
					if statustimes[j][0] == lsampatid and nump.isnan(status[j]).any() == False:
						statuspat = status[j]

				record = nump.hstack((patients[lsampatid],donors[lsamdonid],statuspat,tx_subset.iloc[n,4],tx_subset.iloc[n,5],page,dage))
				survdata = nump.vstack((survdata,record)) #append the observation to the survival data frame
				txtimes = nump.vstack((txtimes,lsamtxtime)) #append the transplant time

			#get rid of the first row that is just zeroes
			survdata = survdata[1:]
			txtimes = txtimes[1:]

			#Compute survival
			values = nump.zeros(nump.shape(survdata)[0])
			for k in range(0, nump.shape(survdata)[0]):
				values[k] = nump.exp(nump.dot(survdata[k],survcoeff))

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
						deaths[m] = 1*(bool(  svalues[m]/365 + txtimes[m] <=maxtime))
						break
					elif svalues[m] < stepsurv[k-1,0] and svalues[m] >= stepsurv[k,0]:
						svalues[m] = stepsurv[k,1]
						#deaths[m] = int(bool( nump.random.uniform(0,1,1) <=stepsurv[k,2]  and svalues[m]/365 + txtimes[m]  <=maxtime))
						deaths[m] = 1*(bool(  svalues[m]/365 + txtimes[m] <=maxtime))
						break

			#Total Deaths
			output_totals.append(nump.sum(deaths))

	return output_totals


def estimate_relist_outcome(directory):
	"""
	This function estimates the number of deaths among relisted patients, those who didn't receive retransplant and those who did.
	Results are written to file in the given directory.
	@Input:
		@directory: directory where the file RawOutput_Relistid.csv, RawOutput_TxIDregraft.csv, and RawOutput_DoIDregraft.csv are
		located. It is also the directory where the output will be written to.
	"""

	#read in relist patients and re-transplanted patients
	relist = pd.read_csv(directory + 'RawOutput_Relistid.csv')
	regraft = pd.read_csv(directory + 'RawOutput_TxIDregraft.csv')

	#obtain avg death and std of death
	waitlist_relist_death = estimate_waitlist_relist_death(relist, regraft)

	#write to csv file
	waitlist_relist_death = pd.DataFrame(waitlist_relist_death)
	waitlist_relist_death.columns = ['Number of Waitlist Relist Deaths']
	waitlist_relist_death.to_csv(directory+'Output_waitlistrelist_deaths.csv')

	#read in donor ids
	doids = pd.read_csv(directory + 'RawOutput_DoIDregraft.csv')

	#estimate post retransplant death
	total_retransplant_death = estimate_post_retransplant_death(regraft, doids)

	#write to csv file
	total_retransplant_death = pd.DataFrame(total_retransplant_death)
	total_retransplant_death.columns = ["Number of Post ReTransplant Death"]
	total_retransplant_death.to_csv(directory + 'Output_post_transplant_deaths_regrafts.csv')










