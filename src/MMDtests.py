import pickle
from time import time 
from math import sqrt, log
from functools import partial 

import torch 
import numpy as np 
import scipy.stats as stats 

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
        Z = torch.cat([X, Y], dim=0)
    dists_ = torch.pdist(Z)
    sig = torch.median(dists_)
    return sig.item()




def get_bootstrap_threshold(X, Y, kernel_func, statfunc, alpha=0.05,
                            num_perms=500, progress_bar=False, device='cuda',
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
        Z = torch.vstack((X, Y))
        Z = Z.to(device)
    # assert len(X)==len(Y)
    n,  n_plus_m = len(X), len(Z)
    # kernel matrix of the concatenated data
    KZ = kernel_func(Z, Z) # 
    
    original_statistic = statfunc(X, Y, kernel_func)
    if use_numpy:
        perm_statistics = np.zeros((num_perms,))
    else:
        perm_statistics = torch.zeros((num_perms,), device=device)

    range_ = tqdm(range(num_perms)) if progress_bar else range(num_perms)
    for i in range_:
        if use_numpy:
            perm = np.random.permutation(n_plus_m)
        else:
            perm = torch.randperm(n_plus_m)
        X_, Y_ = Z[perm[:n]], Z[perm[n:]] 
        stat = statfunc(X_, Y_, kernel_func)
        perm_statistics[i] = stat

    # obtain the threshold
    if use_numpy:
        perm_statistics = np.sort(perm_statistics) 
    else:
        perm_statistics, _ = torch.sort(perm_statistics) 
    i_ = int(num_perms*(1-alpha)) 
    threshold = perm_statistics[i_]
    if not use_numpy:
        threshold = threshold.item()
    if return_stats:
        return threshold, perm_statistics
    else:
        return threshold



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


def runTest(X, Y, kernel_func, stat_func, thresh_func, 
            alpha=0.05, paired=False, thresh_method='bootstrap'):
    # some preprocesseing
    n, m =len(X), len(Y)
    if n!=m and paired:
        if n<m:
            Y = Y[:n]
            m=n
        elif n>m:
            X = X[:m]
            n = m 
    # compute the statistic 
    stat = stat_func(X, Y, kernel_func) 
    # obtain the threshold
    if thresh_method=='bootstrap':
        thresh = thresh_func(X, Y, kernel_func, stat_func, alpha=alpha) 
    elif thresh_method=='spectral':
        thresh = thresh_func(X, Y, kernel_func, alpha=alpha) 
    elif thresh_method=='uniform_convergence':
        thresh = thresh_func(n, m, alpha) 
    elif thresh_method=='normal':
        thresh = thresh_func(alpha)
    # return 1 if stat>thresh else 0
    if stat>thresh:
        return 1 
    else:
        return 0

def runTestLoop(SourceX, SourceY, NN, MM, kernel_func,
                Tests, paired=False, alpha=0.05, num_trials=100):
    #Initailize a dict to save the power for different tests
    Power = {}
    for test_name in Tests:
        Power[test_name] = torch.zeros(NN.shape)

    for i in tqdm(range(num_trials)):
        # set random seeds 
        torch.manual_seed(i)
        np.random.seed(i)

        for j, (n, m) in enumerate(zip(NN, MM)):
            X, Y = SourceX(n), SourceY(m) 
            # run the different tests 
            for test_name, test_info in Tests.items():
                # unpack the information to run the test
                stat_func, thresh_func, thresh_method = test_info 
                # run the test 
                out = runTest(X, Y, kernel_func, stat_func, thresh_func,
                    alpha=alpha, paired=paired, thresh_method=thresh_method)
                # store the result 
                Power[test_name][j] += out 
    # normalize the number of rejections to calculate power
    for test_name in Power:
        Power[test_name] /= num_trials                
    return Power 


def main():
    # the sampling distributions 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d, epsilon = 10, 0.0 # 0.30
    meanX, meanY = torch.ones((d,)),  torch.ones((d,))*(1+epsilon)
    covX, covY = torch.eye(d), torch.eye(d)
   
    def SourceX(n):
        return GaussianVector(mean=meanX, cov=covX, n=n)

    def SourceY(n):
        return GaussianVector(mean=meanY, cov=covY, n=n)


    alpha = 0.05 
    kernel_func = partial(RBFkernel1, bw=sqrt(d))

    n = 500 
    NN = np.linspace(50, n, 50, dtype=int)
    num_trials=20

    # set up function handles for the different statistics 
    unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True) 
    biased_mmd2 = TwoSampleMMDSquared 
    linear_mmd2 = partial(BlockMMDSquared, b=2)
    # block_mmd2 = partial(BlockMMDSquared, b=int(sqrt(n)))
    block_mmd2 = partial(BlockMMDSquared, b=10)
    cross_mmd2 = crossMMD2sampleUnpaired

    #set up function handles for different threshold computing methods
    thresh_bootstrap = partial(get_bootstrap_threshold, 
                                num_perms=num_perms, device=device) 

    thresh_spectral = partial(get_spectral_threshold, numNullSamp=num_perms)

    thresh_normal = get_normal_threshold

    thresh_convergence_biased = partial(get_unifrom_convergence_threshold,
                                            biased=True)
    thresh_convergence_unbiased = partial(get_unifrom_convergence_threshold,
                                            biased=False)

    # set up the quadratic-time test with 
    TESTS = {
        'mmd2-bootstrap': (unbiased_mmd2, thresh_bootstrap, 'bootstrap'), 
        'mmd2-Mcdiarmid': (biased_mmd2, thresh_convergence_biased, 'uniform_convergence'), 
        'mmd2-Hoeffding': (unbiased_mmd2, thresh_convergence_unbiased, 'uniform_convergence'), 
        'mmd2-linear': (linear_mmd2, thresh_bootstrap, 'bootstrap'),
        'mmd2-block': (block_mmd2, thresh_bootstrap, 'bootstrap'), 
        'mmd2-spectral': (unbiased_mmd2, thresh_spectral, 'spectral'), 
        'cross-mmd2': (cross_mmd2, thresh_normal, 'normal')
    }

    Power = runTestLoop(SourceX, SourceY, NN, NN, kernel_func, TESTS, 
                paired=True, alpha=alpha, num_trials=num_trials)
    
    return Power 

def main_new(d=10, epsilon=0.3, n=300, num_trials=200, num_perms=200, alpha=0.05, 
    save_fig=False, save_data=False, figname0=None, figname1=None, 
    filename=None, null=False, RBF=True, num_perturbations=None, plot_figs=True,
    return_data=False, num_steps=20, seed=None, initial_value=10, methods=None
        ):
    # the sampling distributions 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if seed is not None:
        torch.manual_seed(seed)

    if num_perturbations is None:
        # num_perturbations = min(d, 5)
        num_perturbations = d//2
    
    meanX, meanY = torch.ones((d,)),  torch.ones((d,))
    covX, covY = torch.eye(d), torch.eye(d)
    if not null:
        meanY[:num_perturbations] = (1+epsilon)
   
    def SourceX(n):
        return GaussianVector(mean=meanX, cov=covX, n=n)

    def SourceY(n):
        return GaussianVector(mean=meanY, cov=covY, n=n)


    #set up function handles for different threshold computing methods
    thresh_bootstrap = partial(get_bootstrap_threshold, 
                                num_perms=num_perms, device=device) 


    thresh_normal = get_normal_threshold

    assert initial_value<n
    NN = np.linspace(initial_value, n, num_steps, dtype=int)
    block_size = int(sqrt(n))
   
    if methods is None:
        methods = ['c-mmd']

    PowerDict, TimesDict = {}, {}

    for method in methods:
        PowerDict[method] = np.zeros(NN.shape)
        TimesDict[method] = np.zeros(NN.shape)

    for i in tqdm(range(num_trials)):
        for j, ni in enumerate(NN):
            torch.manual_seed(i) 
            X, Y = SourceX(ni), SourceY(ni) 

            bw = get_median_bw(X)
            if RBF:
                kernel_func = partial(RBFkernel1, bw=bw)
            else:
                kernel_func = partial(PolynomialKernel, scale=bw)

            # set up function handles for the different statistics 
            unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True) 
            linear_mmd2 = partial(BlockMMDSquared, b=2)
            cross_mmd2 = crossMMD2sampleUnpaired

            for method in methods:
                start_time = time()
                if method=='mmd-perm':
                    stat = unbiased_mmd2(X, Y, kernel_func)
                    th = thresh_bootstrap(X, Y, kernel_func, unbiased_mmd2, alpha=alpha)
                elif method=='linear-mmd-perm':
                    stat = linear_mmd2(X, Y, kernel_func)
                    th = thresh_bootstrap(X, Y, kernel_func, linear_mmd2, alpha=alpha)
                elif method =='block-mmd-perm':
                    stat = linear_mmd2(X, Y, kernel_func, b=block_size)
                    th = thresh_bootstrap(X, Y, kernel_func, linear_mmd2, alpha=alpha)
                # elif method=='mmd2-spectral':
                #     stat = ni*biased_mmd2(X, Y, kernel_func)
                #     th = thresh_spectral(X, Y, kernel_func,  alpha=alpha)
                elif method=='c-mmd':
                    stat = cross_mmd2(X, Y, kernel_func)
                    th = thresh_normal(alpha)

                PowerDict[method][j] += 1.0*(stat>th)
                TimesDict[method][j] += time() - start_time

    for method in methods:
        PowerDict[method] /= num_trials 
        TimesDict[method] /= num_trials 

    if 'predicted' in methods:
        PowerDict['predicted'] = predict_power(PowerDict['c-mmd'],alpha=alpha)

    if plot_figs:
        palette = sns.color_palette(palette='tab10', n_colors=10)

        # Generate the results dict 
        Results = {}
        Results['num_trials'] = num_trials 
        Results['n'] = n 
        Results['d'] = d 
        Results['epsilon'] = epsilon
        Results['block_size'] = block_size
        Results['num_perturbations'] = num_perturbations
        Results['num_perms'] = num_perms
        Results['PowerDict'] = PowerDict
        Results['TimesDict'] = TimesDict
        Results['methods'] = methods
        Results['palette'] = palette

        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        for i, method in enumerate(methods):
            if method=='predicted':
                ax.plot(NN, PowerDict[method], '--', color=palette[i], label=method)
            else:
                ax.plot(NN, PowerDict[method], color=palette[i], label=method)
        if null:
            ax.set_title("Type-I error vs Sample-Size", fontsize=16)
            ax.set_ylabel('Type-I error', fontsize=14)
        else:
            ax.set_title(f"Power vs Sample-Size (d={d}, $\epsilon$={epsilon})", fontsize=16)
            ax.set_ylabel('Power', fontsize=14)

        ax.set_xlabel('Sample-Size', fontsize=14)
        ax.legend(fontsize=14)
        if save_fig:
            plt.savefig(figname0, dpi=450)
        else:
            plt.show()

        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        for i, method in enumerate(methods):
            if method != 'predicted':
                ax.plot(NN, TimesDict[method], color=palette[i], label=method)
        ax.set_title("Wall Clock Time vs Sample-Size", fontsize=16)
        ax.set_xlabel('Sample-Size', fontsize=14)
        ax.set_ylabel('Wall Clock Time / trial (seconds)', fontsize=14)
        ax.legend(fontsize=14)
        if save_fig:
            plt.savefig(figname1, dpi=450)
        else:
            plt.show()

    if save_data:
        with open(filename, 'wb') as f:
            pickle.dump(Results, f)

    if return_data:
        return PowerDict, TimesDict

if __name__=='__main__':
    # the sampling distributions 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    epsilon = 0.30
    Epsilon = [0.30, 0.2, 0.1]
    dd = [10, 50,  100]
    nn =  [400,  500, 600]

    null = True
    alternative=False
    save_fig = False 
    save_data = False
    alpha = 0.05 
    num_perms=200
    num_trials=100


    for idx, d in enumerate(dd):
        n = nn[idx]
        epsilon = Epsilon[idx]

        start_time = time()
        print('\n' + '-'*50)
        print(f'Starting with d={d}, n={n}')
        if alternative:
            figname0 = f'../data/Expt2_Power_d_{d}_RBF_GMD_new.png'
            figname1 = f'../data/Expt2_Time_d_{d}_RBF_GMD_new.png'
            filename = f'../data/Expt2_d_{d}_n_{n}_RBF_GMD_new.pkl'

            main_new(d=d, n=n, save_fig=save_fig, save_data=save_data,
                figname0=figname0, figname1=figname1, filename=filename, 
                null=False, RBF=True, num_perms=num_perms, num_trials=num_trials,
                epsilon=epsilon)
        elif null:
            figname0 = f'../data/Expt1_Power_d_{d}_RBF_null_GMD_new.png'
            figname1 = f'../data/Expt1_Time_d_{d}_RBF_null_GMD_new.png'
            filename = f'../data/Expt1_d_{d}_n_{n}_RBF_null_GMD_new.pkl'
            main_new(d=d, n=n, save_fig=save_fig, save_data=save_data,
                figname0=figname0, figname1=figname1, filename=filename, 
                null=True, RBF=True, num_perms=num_perms, num_trials=num_trials, 
                epsilon=epsilon)


        time_taken = time() - start_time 
        print('\n')
        print(f'd={d}, n={n} took {time_taken:.2f} seconds') 
        print('\n' + '-'*50)

