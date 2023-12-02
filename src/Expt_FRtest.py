import argparse
import numpy as np 
from scipy.sparse.csgraph import minimum_spanning_tree as mst 
from scipy.spatial.distance import cdist as cdist 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from time import time 

from MMDtests import main

def createGMDSource(epsilon=1.0, d=10):
    meanX = np.zeros((d,))
    meanY = np.zeros((d,))
    meanY[0] = epsilon
    covX, covY = np.eye(d), np.eye(d)
    def Source(n, m=None):
        m = n if m is None else m
        X = np.random.multivariate_normal(mean=meanX, cov=covX, size=n)
        Y = np.random.multivariate_normal(mean=meanY, cov=covX, size=n)
        return X, Y 
    return Source         


################################################################################
################################################################################
def FRstatistic(X, Y):
    n = len(X)
    Z = np.concatenate((X, Y), axis=0)
    D = cdist(Z, Z)
    M = 1.0*(mst(D)>0)
    # now compute the number of runs: 
    R = 1 + np.sum(M[:n, n:]) 
    return R 

def getFRthreshold(statfunc, XX, YY, num_perms=200, alpha=0.05, return_vals=False):
    FRvals = np.zeros((num_perms,))
    n, m = len(XX), len(YY)
    N = n+m
    Z = np.concatenate((XX, YY), axis=0)
    for i in range(num_perms):
        perm = np.random.permutation(N) 
        Z_ = Z[perm] 
        X_, Y_ = Z_[:n], Z_[n:]
        FRvals[i] = statfunc(X_, Y_)
    FRsorted = np.sort(FRvals) 
    threshold = FRsorted[int(num_perms*alpha)] 
    if return_vals:
        return threshold, FRsorted 
    else:
        return threshold
################################################################################
################################################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dims', default=10, type=int,  help='dimension of features')
    parser.add_argument('-n', '--num_samples', default=200, type=int,  help='sample-size')
    parser.add_argument('-ns', '--num_steps', default=10, type=int,  help='number of steps for plotting power curves')
    parser.add_argument('-np', '--num_perms', default=400, type=int,  help='number of permutations for FR test')
    parser.add_argument('-a', '--alpha', default=0.05, type=float,  help='significance level of the test')
    parser.add_argument('-nt', '--num_trials', default=200, type=int,  help='number of repetitions to plot power curves')
    parser.add_argument('-e', '--epsilon', default=1.0,  type=float, help='perturbation for the GMD source')
    parser.add_argument('--save_fig', action='store_true',
                            help="choose whether to save the figures or not")
    args = parser.parse_args() 

    d = args.dims
    epsilon = args.epsilon 
    save_fig=args.save_fig
    n = args.num_samples
    num_steps = args.num_steps
    num_trials = args.num_trials
    alpha = args.alpha
    num_perms = args.num_perms

    # Set the random seed 
    seed = int(time())%9999

    assert n>2
    initial_n = min(n//2, 20)
    NN = np.linspace(initial_n, n, num_steps, dtype=int)

    # Get the power curve 
    SourceAlt = createGMDSource(epsilon=epsilon, d=d)
    PowerFR = np.zeros((num_steps,))

    for trial in tqdm(range(num_trials)):
        for i, n in enumerate(NN):
            X, Y = SourceAlt(n) 
            stat = FRstatistic(X, Y)
            th = getFRthreshold(FRstatistic, XX=X, YY=Y, alpha=alpha,
                                    num_perms=num_perms)
            if stat<th: # rejectj
                PowerFR[i] += 1.0

    PowerFR /= num_trials 
    # get the cross-MMD power 
    PowerDict, TimesDict = main_new(d=d, epsilon=epsilon, n=n, 
                                num_trials=num_trials,
                                alpha=alpha,
                                num_perturbations=1, # same as the GMD-source 
                                return_data=True, 
                                plot_figs=False,
                                num_steps=num_steps,
                                seed=seed,
                                initial_value=initial_n)
    PowerCMMD = PowerDict['c-mmd']
    ###plot the results 
    plt.figure()
    plt.plot(NN, PowerCMMD, label='c-MMD')
    plt.plot(NN, PowerFR, label='FR')
    plt.legend(fontsize=12)
    plt.xlabel('Sample-Size (n)', fontsize=13)
    if epsilon==0:
        plt.ylabel('Type-I error', fontsize=13)
        plt.title(f"GMD Source (d={d}, {num_trials} trials)")
    else:
        plt.ylabel('Power', fontsize=13)
        plt.title(f"GMD Source ( $\epsilon$={epsilon}, d={d})")
    if save_fig:
        figname = '../data/' + f'Power_FR_d_{d}_seed_{seed}.png'
        plt.savefig(figname, dpi=450)
    else:
        plt.show()

