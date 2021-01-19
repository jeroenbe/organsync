#!/sscc/opt/anaconda3/bin/python
import numpy as nump

ndsa = 58 #number of DSAs

###################################################################Simulation Engine Classes#####################################################
class Event:
    """
    This is an Event class.
    Attributes:
        @event_type: type of event
        @event_time: time of event
        @event_information: information of event
    """
    def __init__(self,event_type,event_time,event_information):
        self.type = event_type
        self.time = event_time
        self.info = event_information


class G:
    """
    This is a class that contains all the global variables relevant to the Simulator
    """
    seed = 7777 #Random Number Seed
    maxtime=5  #Maximum Clock Time
    nreps =5   #Number of Replications
    clock =0    #Clock

    #pid = 100000    #Patient id variable, should be larger than size of initial waitlist
    oid = 0         #Donor id variable

    maxrejects =999          #Maximum number of  rejections from COMPATIBLE recipients before discarding organ
    #progtime = 30/365       #Time until  next patient disease progression (non HCC)

    regional_sharing =0     #1=Full Regional Sharing /0=Standard (Share 15 + Share 35)
    sodium =1              #1=Use MELD Sodium
    capanddelay =0          #1=Use HCC cap and delay
    spartners =0            #1=Use Sharing Partners

    ShareU = 35 #threshold for "Share35 Policy"
    ShareL = 15 #threshold for "Share15" Policy

    localboost = 0   #MELD score boost for local patients
    regionalboost = 0 #MELD score boost for regional patients


    # Initialize Output
    record_deaths = nump.zeros(shape=(1,3)) #record number of deaths per year
    record_mr_disparity_mean = nump.zeros(shape=(1,3)) # record DSA average mortality rate per year
    record_mr_disparity_std = nump.zeros(shape=(1,3)) # record standard deviation of DSA mortality rate
    record_meld_disparity_mean = nump.zeros(shape=(1,3)) #record average MELD score at transplant for non-status 1 candidates
    record_meld_disparity_std = nump.zeros(shape=(1,3)) #record standard deviation of MELD score at transplant for non-status 1 candidates
    record_medMELDmean =nump.zeros(shape=(1,3)) #record median MELD at transplant over the year for non-Status 1 candidates
    record_medMELDstd=nump.zeros(shape=(1,3)) #record standard deviation of median MELD at transplant for non-status 1 candidates

    record_ydeaths = nump.zeros(shape=(1,ndsa+3)) #record number of deaths across DSAs
    record_ytransplants=nump.zeros(shape=(1,ndsa+3)) #record number of transplants across DSAs
    record_yarrivals=nump.zeros(shape=(1,ndsa+3)) #record number of patient arrivals across DSAs
    record_ycandidates=nump.zeros(shape=(1,ndsa+3)) #record number of candidates at the beginning of the year across DSAs
    record_yremoved=nump.zeros(shape=(1,ndsa+3)) #record number of waitlist candidates removed from waitlist during the year besides death or transplant 
    record_ywait=nump.zeros(shape=(1,ndsa+3),dtype=float) #record accumulated total transplant waiting time across DSAs
    record_yMELD =nump.zeros(shape=(1,ndsa+3)) #record accumulated total transplant MELD scores of patients across DSA
    record_txDSA = nump.zeros(shape=(ndsa,ndsa)) #record the total number of organs procured from DSA i to DSA j for all years and replications
    record_txDSAoutput = nump.zeros(shape=(ndsa,ndsa)) #record the number of livers from DSA i to DSA j at replication-year(t)
    record_txID = nump.zeros(shape=(1,6)) #record patients who were transplanted; does not include those who were ever or would have been relisted
    record_doID = nump.zeros(shape=(1,5)) #record patients who were transplanted along with their corresponding donors; does not include transplant patients who were ever or would have been relisted
    record_removals = nump.zeros(shape=(1,6)) #record patients removed for any reason besides transplant/death
    record_yrelists = nump.zeros(shape=(1,ndsa+3)) #record number of candidates relisted for transplant during the byear by DSA
    record_yregrafts = nump.zeros(shape=(1,ndsa+3)) #record number of relisted candidates who received re-transplant durint the year by DSA
    record_txIDregraft = nump.zeros(shape=(1,6)) #record patients who were re-transplanted
    record_doIDregraft = nump.zeros(shape=(1,5)) #record patients who were re-transplanted along with corresponding donors
    record_relists = nump.zeros(shape=(1,6)) #record information of patient relisted



class SimStat:
    """
    Simulation Statistics
    """
    numcandidates =nump.zeros(shape=(ndsa,1),dtype=int) #number of candidates across DSA
    ycandidates = nump.zeros(shape=(ndsa,1),dtype=int) #number of candidates at beginning of year across DSA
    yarrivals =   nump.zeros(shape=(ndsa,1),dtype=int) #number of patient arrivals across DSAs
    ydeaths =     nump.zeros(shape=(ndsa,1),dtype=int) #number of deaths across DSAs
    yremoved =     nump.zeros(shape=(ndsa,1),dtype=int) #number of removals across DSAs
    ytransplants = nump.zeros(shape=(ndsa,1),dtype=int) #number of transplant across DSAs
    ywait = nump.zeros(shape=(ndsa,1),dtype=float) #accumulated total waiting time across DSAs
    yMELD = nump.zeros(shape=(ndsa,1),dtype=int) #accumulated total MELD scores of patients across DSAs
    ymedMELD = [[] for i in range(0,ndsa)] #a list of MELD scores for each DSA
    yrelists =     nump.zeros(shape=(ndsa,1),dtype=int) #number of relists across DSAs
    yregrafts =    nump.zeros(shape=(ndsa,1),dtype=int) #number of re-transplants across DSAs


##################################################################################################################################################################################
