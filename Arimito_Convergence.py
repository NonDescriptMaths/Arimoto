#GeomStats imports
import geomstats.backend as gs
from geomstats.information_geometry.categorical import CategoricalDistributions
#Standard imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#Amirito imports
import Arimito_Generator as Generator
import Arimito as arimito


def plot_path_binary(prior= np.array([0.2,0.8]),
                     channel_matrix = Generator.symmetric_channel_matrix(xdim=2,ydim=2,e=0.2),
                     n=1,
                     include_geodesic = False,
                     lang = "Python"):
    '''
    This function will plot the path our convergence takes for a binary channel

    prior: Initial probability distribution
    channel_matrix: Transition matrix for the channel
    n: Number of paths to plot
    include_geodesic: If True, geodesic paths will be plotted from the initial prior to the solution
    '''
    if n == 1:
        prior = [prior]
    else:
        prior = [Generator.random_prior(xdim=2) for i in range(n)]

    fig = plt.figure()
    ax = fig.add_subplot()
    x = np.linspace(0, 1, 100)
    y = 1 - x
    ax.plot(x, y, 'grey',zorder=1)
    for i in range(len(prior)):
        if lang == "C++":
            C,p,p_route = arimito.Carimito(prior[i],channel_matrix)
        else:
            C,p,p_route = arimito.arimito(prior[i],channel_matrix)

        ax.scatter(p_route[:,0],p_route[:,1],s=5 ,zorder=2)
    ax.set_title("Convergence Path for a Binary Channel")
    ax.set_xlabel("P(0)")
    ax.set_ylabel("P(1)")

    if include_geodesic:
        ArgMax = p_route[-1]
        Manifold = CategoricalDistributions(dim = 2)
        times = gs.linspace(0., 1., 100)
        geodesics = []
        for i in range(len(prior)):
            geodesics.append(Manifold.metric.geodesic(initial_point=prior[i], end_point=ArgMax)(times))

        for i in range(len(prior)):
            ax.plot(geodesics[i][:,0], geodesics[i][:,1])
    plt.show()

def plot_path_ternary(prior = np.array([0.8,0.1,0.1]),
                      channel_matrix = Generator.random_channel_matrix(xdim=3,ydim=3),
                      n=1,
                      include_geodesic = False,
                      lang = "Python"):
    '''
    This function will plot the path our convergence takes for a ternary channel

    prior: Initial probability distribution
    channel_matrix: Transition matrix for the channel
    n: Number of paths to plot
    include_geodesic: If True, geodesic paths will be plotted from the initial prior to the solution
    '''
    if n == 1:
        prior = [prior]
    else:
        prior = [Generator.random_prior(xdim=3) for i in range(n)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    ax.plot_trisurf(x, y, z, color='grey', alpha=0.5)

    for i in range(len(prior)):
        if lang == "C++":
            C,p,p_route = arimito.Carimito(prior[i],channel_matrix)
        else:
            C,p,p_route = arimito.arimito(prior[i],channel_matrix)
        ax.scatter(p_route[:,0],p_route[:,1],p_route[:,2],s=1)
    
    ax.view_init(elev=60, azim=70)
    ax.set_title("Convergence Path for a Ternary Channel")
    ax.set_xlabel("P(0)")
    ax.set_ylabel("P(1)")
    ax.set_zlabel("P(2)")
    if include_geodesic:
        ArgMax = p_route[-1]
        Manifold = CategoricalDistributions(dim = 3)

        times = gs.linspace(0., 1., 100)
        for i in range(len(prior)):
            geodesic = Manifold.metric.geodesic(initial_point=prior[i], end_point=ArgMax)(times)
            ax.plot(geodesic[:,0], geodesic[:,1],geodesic[:,2],alpha = 0.7)

    plt.show()

def plot_path_quarterly(prior = np.array([0.25,0.25,0.25,0.25]),
                        channel_matrix = Generator.random_channel_matrix(xdim=4,ydim=4),
                        n=1,
                        include_geodesic = False,
                        lang = "Python"):
    '''
    This function will plot the path our convergence takes for a quarterly channel

    prior: Initial probability distribution
    channel_matrix: Transition matrix for the channel
    n: Number of paths to plot
    include_geodesic: If True, geodesic paths will be plotted from the initial prior to the solution
    '''
    if n == 1:
        prior = [prior]
    else:
        prior = [Generator.random_prior(xdim=4) for i in range(n)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = np.array([[np.sqrt(8/9), -1/3, 0],  # Point 1
                   [-np.sqrt(2/9), -1/3, np.sqrt(2/3)],  # Point 2
                   [-np.sqrt(2/9), -1/3, -np.sqrt(2/3)],  # Point 3
                   [0, 1, 0]])  # Point 4
    vertices = [[points[0], points[1], points[3]],
            [points[0], points[2], points[3]],
            [points[1], points[2], points[3]],
            [points[0], points[1], points[2]]]
    # Plot sides
    ax.add_collection3d(Poly3DCollection(vertices, facecolors='grey', linewidths=1, edgecolors='grey', alpha=.25))
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    for i in range(len(prior)):
        if lang == "C++":
            C,p,p_route = arimito.Carimito(prior[i],channel_matrix)
        else:
            C,p,p_route = arimito.arimito(prior[i],channel_matrix)
        #ax.scatter(p_route[:,0],p_route[:,1],p_route[:,2],p_route[:,3],s=1)
        # Map each component of the points in p_route to a vertex of the tetrahedron
        p_route_3d = np.zeros((p_route.shape[0], 3))
        for j in range(4):
            p_route_3d += np.outer(p_route[:, j], points[j])

        # Plot the 3D points
        ax.scatter(p_route_3d[:, 0], p_route_3d[:, 1], p_route_3d[:, 2], s=1)


    if include_geodesic:
        ArgMax = p_route[-1]
        Manifold = CategoricalDistributions(dim=4)

        t = np.linspace(0, 1, 100)
        for i in range(len(prior)):
            # Compute the geodesic from the prior to the argmax
            geodesic_points = Manifold.metric.geodesic(initial_point=prior[i], end_point=ArgMax)(t)

            # Map each component of the points in p_route and geodesic_points to a vertex of the tetrahedron
            p_route_3d = np.zeros((p_route.shape[0], 3))
            geodesic_points_3d = np.zeros((geodesic_points.shape[0], 3))
            for j in range(4):
                p_route_3d += np.outer(p_route[:, j], points[j])
                geodesic_points_3d += np.outer(geodesic_points[:, j], points[j])
            ax.plot(geodesic_points_3d[:, 0], geodesic_points_3d[:, 1], geodesic_points_3d[:, 2])

    ax.set_title("Convergence Path for a Quarterly Channel")
    plt.show()


def simulate_distribution_binary(n= 20000,ydim=2):
    '''
    Simulate the optimiser for stochastic channel matrices

    '''
    Distributions = []
    fig = plt.figure()
    ax = fig.add_subplot()
    x = np.linspace(0, 1, 100)
    y = 1 - x
    ax.plot(x, y, 'grey',zorder=1)
    for i in range(n):
        channel_matrix = Generator.random_channel_matrix(xdim = 2,ydim=2)
        prior = Generator.random_prior(xdim = 2)
        C,p,p_route = arimito.arimito(prior,channel_matrix)
        Distributions.append(p)
        ax.scatter(p[0],p[1],c = "purple",s =3)
    ax.set_title("Distributions of Solutions for a Binary Channel")
    ax.set_xlabel("P(0)")
    ax.set_ylabel("P(1)")
    plt.show()


def simulate_distribution_quarterly(n=2000, ydim = 4,bins=8):
    Distributions = []
    cmap = plt.get_cmap('viridis')
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    points = np.array([[np.sqrt(8/9), -1/3, 0],  # Point 1
                   [-np.sqrt(2/9), -1/3, np.sqrt(2/3)],  # Point 2
                   [-np.sqrt(2/9), -1/3, -np.sqrt(2/3)],  # Point 3
                   [0, 1, 0]])  # Point 4
    vertices = [[points[0], points[1], points[3]],
            [points[0], points[2], points[3]],
            [points[1], points[2], points[3]],
            [points[0], points[1], points[2]]]
    # Plot sides
    ax.add_collection3d(Poly3DCollection(vertices, facecolors='grey', linewidths=1, edgecolors='grey', alpha=.25))
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    for i in range(n):
        channel_matrix = Generator.random_channel_matrix(xdim = 4,ydim=ydim)
        prior = Generator.random_prior(xdim = 4)
        C,p,p_route = arimito.arimito(prior,channel_matrix,display_results=False)
        p_3d = np.zeros(3)
        
        for j in range(4):
            p_3d += p[j]*points[j]
        Distributions.append(p_3d)

    Distributions = np.array(Distributions)
    H, edges = np.histogramdd(Distributions, bins=bins)

    # Plot each bin
    sizes = H.flat / H.max() * 100  # Adjust the constant to get the desired range of sizes

    for bin in np.argwhere(H > 0):
        point = [edge[bin[i]] for i, edge in enumerate(edges)]
        index = np.ravel_multi_index(bin, H.shape)
        ax.scatter(*point, s=sizes[index], c=cmap(np.array([[H.flat[index] / H.max()]])),alpha = 0.5)

    ax.set_title("Distributions of Solutions for a Binary Channel")
    ax.set_xlabel("P(0)")
    ax.set_ylabel("P(1)")
    ax.set_zlabel("P(2)")
    plt.show()

#Examples:
#
#
# BINARY-------------------------------------------------------
# Binary Symmetric Channel with 1 path, e = 0.1 and no geodesic

#plot_path_binary(n=1)
#-------------------------------------------------------------
#-------------------------------------------------------------
# Binary Symmetric Channel with 3 paths, e = 0.1 and no geodesic

#plot_path_binary(n=3)
#-------------------------------------------------------------
#-------------------------------------------------------------
# Binary non-symmetric random Channel with 1 path, e = 0.1 and no geodesic

#plot_path_binary(channel_matrix = Generator.random_channel_matrix(xdim=2,ydim=2),n=1)
#-------------------------------------------------------------
#-------------------------------------------------------------
# Binary custom channel with 1 path, e = 0.1 and no geodesic

#plot_path_binary(channel_matrix = np.array([[0.5,0.3],[0.5,0.7]]))
#-------------------------------------------------------------

# TERNARY-----------------------------------------------------
# Ternary Symmetric Channel with 1 path, e = 0.1 and a geodesic

#plot_path_ternary(n=1,channel_matrix = Generator.symmetric_channel_matrix(xdim=3,ydim=3,e=0.1),include_geodesic=False)
#-------------------------------------------------------------
#-------------------------------------------------------------
# Ternary Symmetric Channel with 3 paths, e = 0.1 and no geodesic

#plot_path_ternary(n=10,include_geodesic=True)
#-------------------------------------------------------------
#-------------------------------------------------------------

#plot_path_ternary(channel_matrix = Generator.symmetric_channel_matrix(xdim=3,ydim=3,e=0.2),n=3,include_geodesic=True)
    

#plot_path_quarterly(n=1,include_geodesic=True)
    
#simulate_distribution_binary(n=20)

#simulate_distribution_quarterly(n=2000)
    
def Carimito_multiple(log_base: float = 2, 
                    thresh: float = 1e-12, 
                    maxiter: int = 1e3, 
                    n: int = 1):
    '''
    Perform the arimito algorithm n times and output the distribution
    '''

    p_vals = []
    for i in range(n):
        prior = Generator.random_prior(xdim = 3)
        random_binary_channel = Generator.random_channel_matrix(xdim = 3,
                                                                ydim = 3)
        C,p,p_route = arimito.Carimito(prior,
                                random_binary_channel,
                                log_base=log_base,
                                thresh=thresh,
                                maxiter=maxiter,
                                display_results = False)
        if max(p) != 0:
            p_vals.append(p)
    return p_vals

