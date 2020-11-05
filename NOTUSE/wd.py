# D Amrhein, August 2017
# Updated January 2018

import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib

def pproxy_expt(prior,mask,B,H,snr,Nt,Nlags,kt,gamma):
    """
    prior: ndarray
        2D matrix of gridded fields with dimensions space x time
    mask: ndarray
        2D Boolean pseudoproxy mask array with dimensions lat x lon
    Nlags: int
        Number of lags computed. These will be at time scales specified 
        by m*tau_s, m=0:Nt
    Nt: ndarray
        Number of times reconstructed
    kt: int
        Number of EOFs retained in representing the state vector
    """

    #####  Make pseudoproxies  #####

    # Size of state
    L = np.product(mask.shape)

    # Get pproxy obs uncertainty based on constant snr
    sigma = gt_pproxy_sigma_from_SNR(prior,mask,snr)

    # Make pseudoproxy data and unravel into whole-domain
    ypp = mk_pproxies(prior[:,-Nt:],mask,sigma).ravel('C')

    #####  Construct problem projected into EOF space  #####

    # Obtain solution in EOF space
    [xe,yest,Cxxe,Tu,Tve,nest,nestp] = ls_soln(ypp,H,B,sigma,Nt,gamma)

    #####  Project out of EOF space  #####

    xest   = np.dot(eB,xe.reshape(-1,Nt,order='C'))
    
    # Only compute diagonals of Cxx and Tv in full space
    # because they can be very large
    Cxxii = np.zeros(L*Nt)*np.nan
    for ii in range(Nt):
        dblk = Cxxe[ii*kt:(ii+1)*kt,ii*kt:(ii+1)*kt]
        Cxxii[ii*L:(ii+1)*L] = np.sum(np.dot(eB,dblk)*eB,1)
        
    Tvii = np.nan*np.zeros(L*Nt)
    for ii in range(Nt):
        dblk = Tve[ii*kt:(ii+1)*kt,ii*kt:(ii+1)*kt]
        Tvii[ii*L:(ii+1)*L] = np.sum(np.dot(eB,dblk)*eB,1)

    # Compute scaled residuals

    yest = np.dot(np.eye(L)[np.ndarray.flatten(mask,'C'),:],xest)

    return xest, yest, ypp, Cxxii, Tu, Tvii, nest, nestp
    
def ls_soln(y,H,B,sigma,Nt,gamma):
    """
    Solve a row- and column-weighted tapered least squares problem.

    Parameters
    ----------
    y: ndarray
        Vector of observations
    H: ndarray
        Matrix relating proxies to the state. For now this is just input for a single time and tiled by Nt in
        constructing the problem; later this can be changed to allow data speak over multiple times.
    B: ndarray
        2D matrix of prior covariances in space and time
    sigma: ndarray
        Observational uncertainty. If only a single value is given, 
        it will be used for all pseudoproxies at all times.
    Nt: ndarray
        Number of times reconstructed

    Returns
    -------
    xt: ndarray
        Solution vector
    uA: ndarray
        2D left singular vectors of the design matrix
    vA: ndarray
        2D right singular vectors of the design matrix
    sA: ndarray
        1D singular values of the design matrix
    """
        
    #####  Impose row and column scaling ######

    # The SVD of B is used for square roots
    uB,sB,vBt = np.linalg.svd(B.todense(),full_matrices=False)
    # Right square root pinv of B. This is S^-(1/2)' in Carl's book.
    rsqBi = np.asarray(np.dot(np.diag(1/np.sqrt(sB)),uB.T))
    # Right square root of B. This is S^(1/2)' in Carl's book.
    rsqB = np.asarray(np.dot(uB,np.diag(np.sqrt(sB))))

    # Make a vector of sigma values (obs unc) if necessary
    if np.min(np.size(sigma))==1:
        svec = sigma*np.ones(y.size/Nt)
    else:
        svec = sigma

    # Tile obs uncertainty into all space and time
    sveca = np.tile(svec,Nt)

    # Row weight H
    RH = (1./svec*H.T).T

    # Row weight observations
    Ry = y*1./sveca
    
    # Create the problem design matrix (weighted version of H)
    A = np.dot(np.kron(np.eye(Nt),RH),rsqB)
    
    #####  Obtain a solution in scaled coordinates  ######

    [xestp,Cxxp,Tup,Tvp] = tap_LS(Ry,A,gamma)
    
    #####  Un-scale solution, uncertainty, and resolution  #####

    # Unscale solution
    xest = (np.dot(rsqB,xestp))

    # Unscale solution uncertainty
    Cxx = np.linalg.multi_dot([rsqB,Cxxp,rsqB.T])

    # Unscale resolution matrices
    Wsq  = scipy.sparse.spdiags(   sveca.T,0,sveca.size,sveca.size)
    Wsqi = scipy.sparse.spdiags(1./sveca.T,0,sveca.size,sveca.size)

    Tu = np.linalg.multi_dot([Wsq.todense(),Tup,Wsqi.todense()])
    Tv = np.linalg.multi_dot([rsqB,Tvp,rsqBi])

    # yest are the reconstructed values at data locations
    yest = np.linalg.multi_dot([Wsq.todense(),A,xestp])

    # nest are the estimated noise (residuals)
    nest = y - yest

    # nestp are the residuals scaled by prior uncertainty
    nestp = np.dot(Wsqi.todense(),nest.T)

    return xest,yest,Cxx,Tu,Tv,nest,nestp

def mkB(prior,N,NT):
    """
    Constructs a whole-domain covariance matrix from a time series of gridded fields (e.g. from model output).
    Covariances in space and time do not change as a function of time (weak-sense stationary).
    
    Parameters
    ----------
    prior: ndarray
        2D matrix of gridded fields with dimensions space x time
    N: int
        Number of times represented by the covariance matrix
    NT:int
        Number of lags computed (off block-diagonal elements). NT<=N. NT = 1 corresponds to zero-lag.

    Returns
    -------
    B: ndarray
        2D matrix of covariances in space and time. The matrix is block symmetric.

    """

    if NT==0:
        raise Exception('Lag covariances must be computed for at least 1 time, i.e. NT>0')

    if NT>N:
        raise Exception('NT must be <= N')

    # Make the block matrix row of time-lag covariances
    # Initialize
    [lt,kt] = prior.shape
    B = scipy.sparse.kron(np.diag(np.ones(N)),np.zeros([kt,kt]))

    for tau in np.arange(NT):
        # dt is the projected prior set with the last NT years missing. When tau=0, this is d0. 
        # For tau>0, values are shifted forward by tau time increments.
        dt = prior[tau:,:]

        if tau==0: 
            dr = dt
            Nr = dr.shape[0]
            B = B + 1./(Nr)*scipy.sparse.kron(np.diag(np.ones(N-tau), tau),np.dot(dr.T,dt))
        else: 
            dr = prior[:-tau,:]
            Nr = dr.shape[0]
            B = B + 1./(Nr)*scipy.sparse.kron(np.diag(np.ones(N-tau), tau),np.dot(dr.T,dt))
            # Blocks need not be symmetric, but c should!
            B = B + 1./(Nr)*scipy.sparse.kron(np.diag(np.ones(N-tau),-tau),np.dot(dr.T,dt).T)
            print(Nr)
    return B

def gtSkill(fld1,fld2):

    """
    Computes RMSE, Pearson correlation, and CE given true values in fld1 and reconstruted values in fld2.

    Parameters
    ----------
    
    fld1: ndarray
        3D "true" values with dimensions time x lat x lon
    fld1: ndarray
        3D reconstructed values with dimensions time x lat x lon

    Returns
    -------
    rmse: ndarray
        2D Root-mean squared error
    correlation: ndarray
        2D Pearson correlation
    ce: ndarray
        2D Coefficient of efficiency
    """
    l,m,n = fld1.shape

    rmse = np.sqrt(np.nanmean((fld1-fld2)**2.,0))
    
    # Just compute the diagonal of the covariance matrix. 
    # I divide by l-2 (rather than l-1 for an unbiased covariance estimator) because 
    # we can only look at l-1 predicted values.

    # Computing stds by hand because using np.std was giving weird results 
    # (possibly due to the different normalization)

    cvec = np.nansum(
                  (fld1-np.nanmean(fld1,0))
                  *(fld2-np.nanmean(fld2,0))
                  ,0)/(l-1.)
    std1 = (np.nansum(
                  (fld1-np.nanmean(fld1,0))
                  *(fld1-np.nanmean(fld1,0))
                  ,0)/(l-1.))**.5

    std2 = (np.nansum(
                  (fld2-np.nanmean(fld2,0))
                  *(fld2-np.nanmean(fld2,0))
                  ,0)/(l-1))**.5

    corr = cvec/(std1*std2)

    ce = 1.0 - np.sum((fld1-fld2)**2.,0)/np.sum((fld1-np.nanmean(fld1,0))**2.)

    return rmse, corr, ce


def plotSkill(prior,x,nt,lon,lat):

    nlat = lat.shape[0]
    nlon = lon.shape[0]
    # Reshape "true" values and reconstructed
    tr = prior[:,-nt:].T.reshape(nt,nlat,nlon)
    xr = x.T.reshape(nt,nlat,nlon)
    
    rmse,corr,ce = gtSkill(tr,xr)
#    print(np.mean(rmse))
    print('Global mean correlation is '+ str(np.mean(corr)))
    print('Global mean CE is '+ str(np.mean(ce)))
    
    plt1 = plotMap(lon,lat,corr,-1,1,'Correlation')
    plt2 = plotMap(lon,lat,ce,0,1,'CE')
    

def plotMap(lon,lat,fld,vmi,vma,title):

    plt.figure(figsize=(13,6))
    m = Basemap(projection='robin',llcrnrlat=-87,urcrnrlat=81,lon_0=0,\
                    llcrnrlon=0,urcrnrlon=360,resolution='c');

    # draw parallels and meridians.
    parallels = np.arange(-90.,90.,30.)

    # Label the meridians and parallels
    m.drawparallels(parallels,labels=[False,True,True,False])

    # Draw Meridians and Labels
    meridians = np.arange(-180.,181.,30.)
    m.drawmeridians(meridians)
    m.drawmapboundary(fill_color='white')
    
    x,y = np.meshgrid(lon[:], lat[:])
    
    ax = plt.gca()
    masked_array = np.ma.array(fld, mask=np.isnan(fld))
    cmap = matplotlib.cm.viridis
    cmap.set_bad('white',1.0)
    
    im1 = m.pcolormesh(x,y,fld,shading='flat',cmap=cmap,latlon=True,vmin=vmi,vmax=vma)
    m.drawcoastlines();
#    plt.set_cmap('plasma')
    cbar = plt.colorbar(shrink=.7)
    plt.title(title)
    plt.show()

    return plt


def plotMap_nl(lon,lat,fld,title):

    po = plt.figure(figsize=(13,6))
    m = Basemap(projection='robin',llcrnrlat=-87,urcrnrlat=81,lon_0=0,\
                    llcrnrlon=0,urcrnrlon=360,resolution='c');

    # draw parallels and meridians.
    parallels = np.arange(-90.,90.,30.)

    # Label the meridians and parallels
    m.drawparallels(parallels,labels=[False,True,True,False])

    # Draw Meridians and Labels
    meridians = np.arange(-180.,181.,30.)
    m.drawmeridians(meridians)
    m.drawmapboundary(fill_color='white')
    
    x,y = np.meshgrid(lon[:], lat[:])
    
    ax = plt.gca()
    masked_array = np.ma.array(fld, mask=np.isnan(fld))
    cmap = matplotlib.cm.viridis
    cmap.set_bad('white',1.0)
    
    im1 = m.pcolormesh(x,y,fld,shading='flat',cmap=cmap,latlon=True)
    m.drawcoastlines();
#    plt.set_cmap('plasma')
    cbar = plt.colorbar(shrink=.7)
    plt.title(title)
    plt.show()

    return plt


def plotMap_nl_save(lon,lat,fld,title,fname):

    plt.figure(figsize=(13,6))
    m = Basemap(projection='robin',llcrnrlat=-87,urcrnrlat=81,lon_0=0,\
                    llcrnrlon=0,urcrnrlon=360,resolution='c');

    # draw parallels and meridians.
    parallels = np.arange(-90.,90.,30.)

    # Label the meridians and parallels
    m.drawparallels(parallels,labels=[False,True,True,False])

    # Draw Meridians and Labels
    meridians = np.arange(-180.,181.,30.)
    m.drawmeridians(meridians)
    m.drawmapboundary(fill_color='white')
    
    x,y = np.meshgrid(lon[:], lat[:])
    
    ax = plt.gca()
    masked_array = np.ma.array(fld, mask=np.isnan(fld))
    cmap = matplotlib.cm.viridis
    cmap.set_bad('white',1.0)
    
    im1 = m.pcolormesh(x,y,fld,shading='flat',cmap=cmap,latlon=True)
    m.drawcoastlines();
#    plt.set_cmap('plasma')
    cbar = plt.colorbar(shrink=.7)
    plt.title(title)
    plt.savefig(fname,transparent=True)
    plt.show()


    return plt


def gt_pproxy_sigma_from_SNR(prior,mask,snr):

# NB ravel works with C indexing, not F.

    globalStds = np.std(prior,1)
    sigmas = globalStds[mask.ravel('C')]*1./snr

    return sigmas
    
def mk_pproxies(prior,mask,sigma):
    # Number of obs
    M = np.sum(np.int64(mask))
    # Size of state
    L = np.product(mask.shape)
    
    # Make a vector of sigma values (obs unc) if necessary
    if np.min(np.size(sigma))==1:
        svec = sigma*np.ones([M])
    else:
        svec = sigma

    # Set a seed for consistent pseudoproxy experiments:
    #np.random.seed(0)

    # Extract time series from prior
    ynn = prior[mask.ravel('C')]
    # Make noise that is scaled by svec and add to ynn
    yur = ynn + (svec*np.random.randn(ynn.shape[1],ynn.shape[0])).T

    return yur

def tap_LS(y,A,gamma=1):

    """
    Tapered least-squares solver for problem of the form y=Ax+n. Scalings should have already
    been applied to x and n.
    
    Parameters
    ----------
    
    A describes the inverse problem
    y are the data
    The square of gamma weights the x.T*x in the cost function
    
    Returns
    -------

    H is useful for calculating the uncertainty and resolution matrices
    xest is the estimate of the parameter x
    
    Refs: Wunsch (2006) Eqns (2.120), (2.121), (2.346), (2.350)

    """

    [m,n] = A.shape

    H    = np.linalg.pinv(np.dot(A.T,A)+gamma**2*np.eye(n))
    xest = np.linalg.multi_dot([H,A.T,y])
    Cxx  = np.linalg.multi_dot([H,A.T,A,H.T])
    Tv   = np.linalg.multi_dot([H,A.T,A])
    Tu   = np.linalg.multi_dot([A,H,A.T])

    return xest, Cxx, Tu, Tv

