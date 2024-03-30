import numpy as np
import CArimito
import Arimito_Generator as Generator

#Arimito Algorithm
#---------------------------------------
def arimito(prior,
            channel_matrix,
            log_base: float = 2, 
            thresh: float = 1e-12, 
            maxiter: int = 1e3, 
            display_results = True):
    '''
    Consider I(X,Y;P), where P is the channel matrix, i.e P_ij = p(y=i|x=j)
    We attempt to maximise I with the blahut amirito algorithm
    --------------------------------------------------------
    prior: Inital guess for maximal X
    channel_matrix: is the matrix representing the transition from X to Y, i.e P
    log_base: base of logarithm, typically 2 or nats
    thresh: threshold to finish algorithm, when iterations are < thresh apart
    max_iter: maximum number of iterations before giving up
    --------------------------------------------------------
    BA is a maxmax algorithm, each maximisation has a closed form expression
    We perform the maximisation as a for loop
    '''
    if display_results:
        print("Arimoto: PYTHON EDITION")
    #Check we have a valid dimension for the prior
    assert prior.shape[0] > 1
    #Checkl prior is a valid probability distribution
    assert np.abs(prior.sum() - 1) < 1e-6
    #Check we have a valid dimension for the channel matrix
    n,m = channel_matrix.shape
    m_1 = prior.shape[0]
    assert m == m_1
    #Initialise the prior and Phi
    p = prior
    p_route = np.array([p])
    P = channel_matrix
    W = np.zeros((m,n))
    if display_results:
        print(f"Prior for p(x): {p}")
        print(f"Channel matrix p(y|x): \n {channel_matrix}")

    for iter in range(int(maxiter)):
        q = np.zeros(m)
        #Maximise I(p,W;P) over W
        for i in range(n):
            for j in range(m):
                W[j][i] = (P[i,j]*p[j])/np.dot(P[i,:],p)
            
        #Maximise I(p,W;P) over p
        r = np.zeros(m)
        for j in range(m):
            r[j] = np.exp(np.dot(P[:,j],np.log(W[j,:])))
        for i in range(m):
            q[i] = r[i]/np.sum(r)

        #Add to array of p values
        p_route = np.append(p_route,[q],axis=0)

        #Check if we have converged
        if np.linalg.norm(q-p) < thresh:
            break
        p = q

    #Calculate the capacity
    C = 0
    for i in range(n):
        for j in range(m):
            if p[j] > 0:
                C += p[j]*P[i][j]*np.log(W[j][i]/p[j])
    C = C/np.log(log_base)
    if display_results:
        print('Max Capacity: ', C)
        print('ArgMax: ', p)
        print("-------------------")
        print("-------------------")
    return C,p,p_route
#---------------------------------------
#


def Carimito(prior,
             channel_matrix,
             log_base: float = 2, 
             thresh: float = 1e-12, 
             maxiter: int = 1e3, 
             display_results = True):
    '''
    Consider I(X,Y;P), where P is the channel matrix, i.e P_ij = p(y=i|x=j)
    We attempt to maximise I with the blahut amirito algorithm
    This is a wrapper for the C implementation of the Arimito algorithm, it is dependent on the CArimito module
    --------------------------------------------------------
    prior: Inital guess for maximal X
    channel_matrix: is the matrix representing the transition from X to Y, i.e P
    log_base: base of logarithm, typically 2 or nats
    thresh: threshold to finish algorithm, when iterations are < thresh apart
    max_iter: maximum number of iterations before giving up
    --------------------------------------------------------
    BA is a maxmax algorithm, each maximisation has a closed form expression
    We perform the maximisation as a for loop
    '''
    if display_results:
        print("Arimoto: C EDITION")
    #Check we have a valid dimension for the prior
    assert prior.shape[0] > 1
    #Checkl prior is a valid probability distribution
    assert np.abs(prior.sum() - 1) < 1e-6
    #Check we have a valid dimension for the channel matrix
    n,m = channel_matrix.shape
    m_1 = prior.shape[0]
    assert m == m_1
    if display_results:
        print(f"Prior for p(x): {prior}")
        print(f"Channel matrix p(y|x): \n {channel_matrix}")
    prior_list = prior.tolist()
    channel_matrix_list = channel_matrix.tolist()
    p,C,p_route = CArimito.arimito(channel_matrix_list,prior_list,1000,1e-12)
    p = np.array(p)
    p_route = np.array(p_route)
    if display_results:    
        print('Max Capacity: ', C)
        print('ArgMax: ', p)
        print("-------------------")
        print("-------------------")
    return C,p,p_route


    
#Test Simulations
#---------------------------------------
def binary_symmetric_channel_simulation(prior = np.array([0.5,0.5]),e=0.5):
    '''
    Simulate a binary symmetric channel
    '''
    print("--||Binary Symmetric Channel Simulation||--")
    print("-------------------")
    P = Generator.symmetric_channel_matrix(xdim=2,ydim=2,e=e)
    C_Analytic = 1 - (- e * np.log2(e) - (1-e) * np.log2(1-e))
    C, p,p_route = arimito(prior,channel_matrix=P,display_results=False)
    print("Python Capacity:",C)
    print("Python ArgMax:",p,"::",type(p))
    prior_list = prior.tolist()
    channel_matrix_list = P.tolist()
    #C_1,p_1,p_route1 = Carimito(prior,channel_matrix=P, display_results= False)
    #print("C++ Capacity:",C_1)
    #print("C++ ArgMax:",p_1,"::",type(p_1))
    print("True Capacity: ", C_Analytic)
    print("-------------------")


def ternary_symmetric_channel_simulation(prior = np.array([0.2,0.2,0.6]),e=0.2):
    '''
    Simulate a ternary symmetric channel
    '''
    print("--||Ternary Symmetric Channel Simulation||--")
    print("-------------------")
    P = Generator.symmetric_channel_matrix(xdim=3,ydim=3,e=e)
    C_Analytic = np.log2(3) - (- e * np.log2(e) - (1-e) * np.log2(1-e)) - e
    C, p, p_route = arimito(prior,P)
    print("True Capacity: ", C_Analytic)
    print("-------------------")


def random_channel_simulation(prior = np.array([0.2,0.2,0.6]),xdim=3,ydim=3):
    P = Generator.random_channel_matrix(xdim=xdim,ydim=ydim)
    C,p,p_route = arimito(prior,P)
    print('True Capacity: ', 'unknown')
    return C,p,p_route

def custom_channel_simulation(prior = np.array([[0.2,0.2,0.6]])):
    P = np.array([[0.1,0.3,0.6],[0.8,0.4,0.2],[0.1,0.3,0.2]])
    C,p,p_route = arimito(prior,P)
    print('True Capacity: ', 'unknown')
    return C,p,p_route

def erase_channel_simulation(prior = np.array([0.2,0.8]),e=0.2):
    P = Generator.erasure_channel_matrix(xdim=2,e=e)
    C,p,p_route = arimito(prior,P)
    print('True Capacity: ', 1-e)
    print("-------------------")


#---------------------------------------
## Examples
#---------------------------------------
    
#binary_symmetric_channel_simulation(prior=np.array([0.2,0.8]))

#ternary_symmetric_channel_simulation(prior = np.array([0.2,0.8]),e=0.2)

#erase_channel_simulation()

#binary_symmetric_channel_simulation(prior=np.array([0.2,0.8]),e=0.1)
    
