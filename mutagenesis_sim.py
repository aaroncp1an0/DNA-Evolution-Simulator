#from scipy.interpolate import spline
#from scipy.interpolate import interp1d as Uspline

from matplotlib.mlab import PCA
#import tsne
import random

import numpy as np
#from scipy.special import betaln
import matplotlib.pyplot as plt
from scipy import linalg

import matplotlib.pyplot as plt
import numpy as np
import sys

#%matplotlib inline
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300 #300
#%reload_ext autoreload


########################################################################################################################
###############code start
############################


#this function carries out the 'randomization' given an empty matrix Mi and a subsample (bottleneck population)
#the matrix Mi is repopulated via random sampling of the bottleneck population (sample)
def randomize(Mi,sample, Rmut, Psize, mutation_numbers):
    #Rmut = mutation rate, samples numbers of mutations poisson wise
    #Psize= protein size/binary size vector
    #mutation number = mutations distribution to be handed out to each new protein in Mi
    
    #select random integers to pull from sample, from which to populate Mi
    #we need Psize random integers between size 0 and len(sample)
    populate_Mi = np.random.randint(len(sample), size=[len(Mi)])
    #randomly repopulates Mi pulling from the variants in the bottlneck 'sample'
    Mi = sample[populate_Mi]
    
    #
    INDEXES_samplei = np.repeat(np.array(range(len(mutation_numbers))), mutation_numbers)
    INDEXES_positioni = np.random.randint(Psize,size=[len(INDEXES_samplei)])
    
    #invert (mutate) all positions which are selected for mutation
    #Arry slices down INDEXES_samplei(sample #s which are mutated) and INDEXES_positioni(positions in samplei which are mutated)
    Mi[INDEXES_samplei, INDEXES_positioni] *= False
    
    #returns the new population Mi, which results from creating a new population using 'sample' and mutating it
    return Mi

#this function carries out the evolution of a population
#returns a large matrix which consists of each generation concatenated together
def evolve_forward(Nbottle=1, Ngen=5, Npop=int(1E5), Rmut=1, Psize=int(1E4)):
    #Nbottle=number of items to propogate to each round of evolution
    #Ngen=number of rounds of sampling/mutation to go through
    #Npop=size of population to set at
    #Rmut=rate of mutations, expected value per sequence; actually values draw from exponential distribution
    #Psize=length of 'protein' in bits
    
    #initialize population #dtype='b'  all zeros (actual type False, we are using bool to save memory)
    #this array is very large - it contains ALL of the samples across the Ngen+1 generations x Pop size x Protein size
    M = np.ones([Ngen+1, Npop, Psize], dtype='bool')
    
    for i in range(1,Ngen+1):
        
        #select Nbottle representatives from the population (at time 0 this is just an array of all False)
        #sample_indexes are the random samples which consist of the bottleneck sequences
        sample_indexes = np.random.randint(Npop,size=[Nbottle])
        
        #this slices the matrix Mi at the previous timestep and pulls out the random bottleneck sample (determined by,
        # 'sample_indexes'. At time step zero the entire 'sample' will be values of False.
        sample = M[i-1,sample_indexes]
        
        #this generates an array of numbers which or sampled from poisson distribution
        #each mutation number will then be used to mutate that number of positions in each protein in the new population
        mutation_numbers = np.random.poisson(lam=Rmut,size=Npop)
        
        #generate a new population from the previous sample
        #overwrite the population that currently occupies that position
        #this is calling above function
        M[i] = randomize(M[i], sample, Rmut, Psize, mutation_numbers)
        
    #reshape the total array so it is MxN
    M = np.reshape(M[1:], [Ngen*Npop, Psize])
    #change dtype to int, I am not sure if there is necessary for PCA
    M = M.astype('int')
    #invert values from 0,1 to -1,1. Also this is unnecessary
    M[M==0]=-1
    #return matrix consisting of the evolutionary history
    return M


def radial_plot(PCA_M, N=3, Npop=2, ONEcolor='black', TWOcolor='red', neg_mode=False):
    #
    #PCA_M is a PCA matrix
    #N is number of components to plot
    #Npop is the number of distinct populations embedded in PCA_M (can be 1 or 2)
    #color of the first PCA value
    #neg_mode -> whether to plot the abs value of each component or the negative values too (360 vs. 180 plot)
    
    xN=np.shape(PCA_M)[0]
    
    theta=np.arange(0,np.pi,np.pi/N)
    thetaspread=np.arange(0,np.pi/N,np.pi/N/xN*2)
    
    #print(len(theta), len(thetaspread), xN, len(PCA_M[xN/2:,1]))
    
    ax = plt.subplot(111, projection='polar')
    #ax.set_ylim([-10,10])
    #ax = plt.subplot(111)
    
    for i in range(N): #,4,6,8,10,12,14,16,18]:
        #ax.scatter(results.Y[xN/2:,i]*np.cos(theta[i]+thetaspread), results.Y[xN/2:,i]*np.sin(theta[i]+thetaspread),s=.05, color='black')
        if neg_mode: ax.scatter(theta[i]+thetaspread+(PCA_M[xN/2:,i]<0)*np.pi, 
                                np.abs(PCA_M[xN/2:,i]),s=.05, color=ONEcolor)
            
        else: ax.scatter(theta[i]*2+thetaspread*2, 
                                np.abs(PCA_M[xN/2:,i]),s=.05, color=ONEcolor)
        #ax.scatter(theta[i]+thetaspread, np.abs(PCA_M[xN/2:,i]),s=.05, color=ONEcolor)
    
    if Npop==2: nextColor=TWOcolor
    else: nextColor=ONEcolor
        
    for i in range(N): #,4,6,8,10,12,14,16,18]:    
        #ax.scatter(results.Y[:xN/2,i]*np.cos(theta[i]+thetaspread), results.Y[:xN/2,i]*np.sin(theta[i]+thetaspread),s=.05, color='red')
            
        if neg_mode: ax.scatter(theta[i]+thetaspread+(PCA_M[:xN/2,i]<0)*np.pi, 
                                np.abs(PCA_M[:xN/2,i]),s=.05, color=nextColor)
            
        else: ax.scatter(theta[i]*2+thetaspread*2, 
                                np.abs(PCA_M[:xN/2,i]),s=.05, color=nextColor)
            
        #ax.scatter(theta[i]+thetaspread+(PCA_M[:xN/2,i]<0)*np.pi, np.abs(PCA_M[:xN/2,i]),s=.05, color=TWOcolor)
        #ax.scatter(theta[i]+thetaspread, np.abs(PCA_M[:xN/2,i]),s=.05, color=TWOcolor)
        pass

    else: pass
    #ax.scatter([0,0,0], [4,-9,-4])
    #ax.set_rmax(40)
    #ax.set_rticks([20,40])  # less radial ticks
    #ax.set_rlabel_position(-22.5)
    if neg_mode: ax.set_xticks(np.arange(0,2*np.pi,np.pi/N))
    else: ax.set_xticks(np.arange(0,2*np.pi,np.pi/N*2))
    
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    try: plt.show()
    except: ax.show()
    return


####IMPORT ANY SYS VARIABLES
try: 
	Nbottle1, Nbottle2, Ngen = sys.argv[1], sys.argv[2], sys.argv[3]
	Rmut, Psize, Npop = sys.argv[3], sys.argv[5], sys.argv[6]
	#number of components
	Ncomp = sys.argv[7]
except: 
	print('using defaults')
	#details for simulation
	Nbottle1, Nbottle2, Ngen, Npop = 1, 100, 4, 300
	Rmut, Psize = 1, 400
	#number of components
	Ncomp = 2
	


########################################################################################################################
###############EXAMPLE RUNS

#M1, M2 is a the 'evolution history matrix' give the specified parameters
#the parameters Npop>500 starts to slow down the computer and Psize>100 also slows down the computer
M1 = evolve_forward(Nbottle=1, Ngen=4, Npop=300, Rmut=1, Psize=400)
M2 = evolve_forward(Nbottle=100, Ngen=4, Npop=300, Rmut=1, Psize=400)

#we reshape M1 and M2 to combine them into one matrix for PCA purposes
M = np.reshape([M1,M2],[np.shape(M1)[0]*2, np.shape(M1)[1]])

#run PCA on the join evolutionary histories matrix
results=PCA(M)

########################################################################################################################
###############EXAMPLE RUNS

#this will plot the output of the PCA, results.Y (the reoriented coordinate matrix)
radial_plot(results.Y,N=Ncomp)

    












        
