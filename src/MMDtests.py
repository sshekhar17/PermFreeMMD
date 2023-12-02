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
from MMDutils import * 

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


def main(d=10, epsilon=0.3, n=300, num_trials=200, num_perms=200, alpha=0.05, 
    save_fig=False, save_data=False, figname0=None, figname1=None, 
    filename=None, null=False, RBF=True, num_perturbations=None, plot_figs=True,
    return_data=False, num_steps=20, seed=None, initial_value=10, methods=None, 
    mode=1, num_pts = 50):
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

            bw = median_bw_selector(SourceX, SourceY, X, Y, mode=mode, num_pts=num_pts)

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
    num_perms=20
    num_trials=10


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

            main(d=d, n=n, save_fig=save_fig, save_data=save_data,
                figname0=figname0, figname1=figname1, filename=filename, 
                null=False, RBF=True, num_perms=num_perms, num_trials=num_trials,
                epsilon=epsilon)
        elif null:
            figname0 = f'../data/Expt1_Power_d_{d}_RBF_null_GMD_new.png'
            figname1 = f'../data/Expt1_Time_d_{d}_RBF_null_GMD_new.png'
            filename = f'../data/Expt1_d_{d}_n_{n}_RBF_null_GMD_new.pkl'
            main(d=d, n=n, save_fig=save_fig, save_data=save_data,
                figname0=figname0, figname1=figname1, filename=filename, 
                null=True, RBF=True, num_perms=num_perms, num_trials=num_trials, 
                epsilon=epsilon)


        time_taken = time() - start_time 
        print('\n')
        print(f'd={d}, n={n} took {time_taken:.2f} seconds') 
        print('\n' + '-'*50)

