import pickle
import torch 
from time import time 
from math import sqrt, log
from functools import partial 

import numpy as np 
import scipy.stats as stats 
from scipy.spatial.distance import pdist 

from tqdm import tqdm 
import matplotlib.pyplot as plt 
plt.style.use('seaborn-white')
import seaborn as sns

from utils import * 

def get_median_bw(Z=None, X=None, Y=None):
    """
    Return the median of the pairwise distances (in terms of L2 norm). 
    """
    if Z is None:
        assert (X is not None) and (Y is not None)
        Z = np.concatenate([X, Y], axis=0)
    dists_ = pdist(Z)
    sig = np.median(dists_)
    return sig

def median_bw_selector(SourceX, SourceY, X, Y, mode=1, num_pts=None):
    """
        SourceX: function handle for generating X
        SourceY: function handle for generating X
        X,Y:     nxd arrays of observations
        mode=1: generate num_pts (X, Y) points from SourceX, SourceY
                for median_calculation
        mode=2: Choose the first n/2 points of X and Y for median 
                calculation
    """
    if mode==1: 
        # choose a default value of num_points if needed
        num_pts = 25 if num_pts is None else num_pts 
        # generate a new set of observations for bandwidth selection
        X_, Y_ = SourceX(num_pts), SourceY(num_pts)
    elif mode==2: 
        assert X is not None and Y is not None 
        n, m = len(X), len(Y) 
        # use the first half of the given data for bandwidth selection
        X_, Y_ = X[:n//2], Y[m//2]
    else: 
        raise Exception(f"mode must either be 1 or 2: input = {mode}")
    bw = get_median_bw(X=X_, Y=Y_)
    return bw 
        

def get_bootstrap_threshold(X, Y, kernel_func, statfunc, alpha=0.05,
                            num_perms=500, progress_bar=False,
                            return_stats=False, use_numpy=False):
    """
        Return the level-alpha rejection threshold for the statistic 
        computed by the function handle stat_func using num_perms 
        permutations. 
    """
    assert len(X.shape)==2
    # concatenate the two samples 
    if use_numpy:
        Z = np.vstack((X,Y))
    else:
        Z = np.vstack((X, Y))
    # assert len(X)==len(Y)
    n,  n_plus_m = len(X), len(Z)
    # kernel matrix of the concatenated data
    KZ = kernel_func(Z, Z) # 
    
    original_statistic = statfunc(X, Y, kernel_func)
    if use_numpy:
        perm_statistics = np.zeros((num_perms,))
    else:
        perm_statistics = np.zeros((num_perms,))

    range_ = tqdm(range(num_perms)) if progress_bar else range(num_perms)
    for i in range_:
        if use_numpy:
            perm = np.random.permutation(n_plus_m)
        else:
            perm = np.random.permutation(n_plus_m)
        X_, Y_ = Z[perm[:n]], Z[perm[n:]] 
        stat = statfunc(X_, Y_, kernel_func)
        perm_statistics[i] = stat

    # obtain the threshold
    if use_numpy:
        perm_statistics = np.sort(perm_statistics) 
    else:
        perm_statistics  = np.sort(perm_statistics) 
    i_ = int(num_perms*(1-alpha)) 
    threshold = perm_statistics[i_]
    if not use_numpy:
        threshold = threshold
    if return_stats:
        return threshold, perm_statistics
    else:
        return threshold


def get_normal_threshold(alpha):
    return stats.norm.ppf(1-alpha)

def get_spectral_threshold(X, Y, kernel_func, alpha=0.05, numEigs=None,
                            numNullSamp=200):
    n = len(X)
    assert len(Y)==n

    if numEigs is None:
        numEigs = 2*n-2
    numEigs = min(2*n-2, numEigs)

    testStat = n*TwoSampleMMDSquared(X, Y, kernel_func, unbiased=False)

    #Draw samples from null distribution
    Z = np.vstack((X, Y))
    # kernel matrix of the concatenated data
    KZ = kernel_func(Z, Z) # 
    
    H = np.eye(2*n) - 1/(2*n)*np.ones((2*n, 2*n))
    KZ_ = np.matmul(H, np.matmul(KZ, H))


    kEigs = np.linalg.eigvals(KZ_)[:numEigs]
    kEigs = 1/(2*n) * abs(kEigs); 
    numEigs = len(kEigs);  

    nullSampMMD = np.zeros((numNullSamp,))

    for i in range(numNullSamp):
        samp = 2* np.sum( kEigs * (np.random.randn(numEigs))**2)
        nullSampMMD[i] = samp

    nullSampMMD  = np.sort(nullSampMMD)
    threshold = nullSampMMD[round((1-alpha)*numNullSamp)]
    return threshold


def get_spectral_threshold_torch(X, Y, kernel_func, alpha=0.05, numEigs=None,
                            numNullSamp=200):
    n = len(X)
    assert len(Y)==n

    if numEigs is None:
        numEigs = 2*n-2
    numEigs = min(2*n-2, numEigs)

    testStat = n*TwoSampleMMDSquared(X, Y, kernel_func, unbiased=False)

    #Draw samples from null distribution
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Z = torch.vstack((X, Y))
    Z = Z.to(device)
    # kernel matrix of the concatenated data
    KZ = kernel_func(Z, Z) # 
    
    H = torch.eye(2*n) - 1/(2*n)*torch.ones((2*n, 2*n))
    H = H.to(device)
    KZ_ = torch.mm(H, torch.mm(KZ, H))


    kEigs = torch.linalg.eigvals(KZ_)[:numEigs]
    kEigs = 1/(2*n) * abs(kEigs); 
    numEigs = len(kEigs);  

    nullSampMMD = torch.zeros((numNullSamp,))

    for i in range(numNullSamp):
        samp = 2* torch.sum( kEigs * (torch.randn((numEigs,), device=device))**2)
        nullSampMMD[i] = samp

    nullSampMMD, _ = torch.sort(nullSampMMD)
    threshold = nullSampMMD[round((1-alpha)*numNullSamp)]
    return threshold.item()



def get_unifrom_convergence_threshold(n, m, k_max=1.0, alpha=0.05, biased=False): 
    assert 0<alpha<1
    if biased: 
        #use Mcdiarmid's inequality based bound stated in Corollary 9 of 
        # Gretton et al. (2012), JMLR
        threshold = sqrt(k_max/n + k_max/m)*(1+ sqrt(2*log(1/alpha)))
    else:
        # use Hoeffding's inequality based bound stated in Corollary 11 of 
        # Gretton et al. (2012), JMLR
        threshold = (sqrt(2)*4*k_max/sqrt(m+n)) * sqrt(log(1/alpha))
    return threshold


def runCMMDtest(SourceX, SourceY, n, m, kernel_func=None, 
                    alpha=0.05, num_trials=100, return_stat_vals=False, 
                    mode=1, num_pts=25):
    th = get_normal_threshold(alpha=alpha)
    stat = np.zeros((num_trials,))
    rejected = np.zeros((num_trials,))
    for i in range(num_trials): 
        # get the samples 
        X, Y = SourceX(n), SourceY(m)
        # check if the kernel function is provided
        if kernel_func is None: 
            bw = median_bw_selector(SourceX, SourceY, X, Y, mode=mode, num_pts=num_pts)
            kernel_ = partial(RBFkernel1, bw=bw)
        else:
            kernel_ = kernel_func 
        # compute the statistic 
        stat[i] = crossMMD2sampleUnpaired(X, Y, kernel_func=kernel_)
        # check if stat[i] is above the rejection threshold or not 
        if stat[i]>th:
            rejected[i] = 1
        
    power = np.sum(rejected) / num_trials 
    if return_stat_vals: 
        return stat 
    else:
        return power 


def runCMMDexperiment(SourceX, SourceY, 
                        n, m=None, 
                        kernel_func=None, 
                        num_trials=200,
                        alpha=0.05,
                        num_steps=20,
                        seed=None,
                        initial_value=20, 
                        mode=1, 
                        num_pts=25
):
    if seed is not None: 
        np.random.seed(seed)
    # GMD Source 
    m = n if m is None else m 
    NN = np.linspace(initial_value, n, num_steps, dtype=int)
    MM = np.linspace(initial_value, m, num_steps, dtype=int)
    Power = np.zeros((len(NN), ))
    for i, (n_, m_) in enumerate(zip(NN, MM)):
        power = runCMMDtest(SourceX, SourceY, n_, m_,
                    kernel_func=kernel_func, alpha=alpha,
                    num_trials=num_trials, return_stat_vals=False, 
                    mode=mode, num_pts=num_pts)
        Power[i] = power 
    return NN, MM, Power