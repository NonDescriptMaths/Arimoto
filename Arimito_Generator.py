import numpy as np

'''
A channel is defined by its transition matrices.
P_ij = P(Y = j | X = i) where X is the input and Y is the output.
'''

# Functions to generate channel matrices
#---------------------------------------
def symmetric_channel_matrix(xdim=3,ydim=3,e=0.5):
    '''
    Generate a channel matrix with n rows and m columns and crossover probability e
    '''
    
    P = np.zeros((xdim,ydim))
    for i in range(xdim):
        for j in range(ydim):
            if i == j:
                P[j][i] = 1 - e
            else:
                P[j][i] = e/(ydim-1)
    return P

def random_channel_matrix(xdim=3,ydim=3):
    '''
    Generate a random channel matrix with n rows and m columns
    '''
    P = np.random.rand(xdim,ydim)
    for i in range(xdim):
        P[i] = P[i]/np.sum(P[i])
    return np.transpose(P)

def erasure_channel_matrix(xdim=2,e=0.2,epsilson=0.0000001):
    ydim = xdim +1
    '''
    Generate a deletion channel matrix with n rows and deletion  probability e
    xdim: number of input symbols
    e: deletion probability
    epsilson: small value to avoid division by zero and log0
    '''
    P = np.zeros((xdim,ydim))
    P[0][0] = 1 - e
    P[0][1] = epsilson
    P[1][0] = epsilson
    P[1][1] = 1 - e
    P[0][2] = e
    P[1][2] = e
    return(np.transpose(P))


Custom_channel_matrix = np.array([[0.6,0.3,0.1],[0.7,0.1,0.2],[0.5,0.05,0.45]])
#---------------------------------------

#Function to generate Prior
#---------------------------------------
def uniform_prior(xdim=3):
    '''
    Generate a uniform prior with n elements
    '''
    p = np.ones(xdim)
    return p/np.sum(p)

def random_prior(xdim=3):
    '''
    Generate a random prior with n elements
    '''
    p = np.random.rand(xdim)
    return p/np.sum(p)

#---------------------------------------
