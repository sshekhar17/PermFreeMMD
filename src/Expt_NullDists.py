import argparse
import pickle
from datetime import datetime 
from math import sqrt, log
from functools import partial 

import torch 
import numpy as np 
import scipy.stats as stats 

from tqdm import tqdm 
import matplotlib.pyplot as plt

from MMDutils import get_median_bw
plt.style.use('seaborn-white')
import seaborn as sns

from utils import * 

import tikzplotlib as tpl 

def main(n=400, m=500, d=10, d_high=500, SourceX=None, SourceX2=None,
            save_fig=False, figname=None, num_trials=2000, kernel_type='RBF',
            save_data=False, filename=None, poly_degree=2):

    # default sources are Gaussian 
    if SourceX is None:
        def SourceX(n):
            return GaussianVector(mean=torch.zeros((d,)), cov=torch.eye(d), n=n)
    if SourceX2 is None:
        def SourceX2(n):
            return GaussianVector(mean=torch.zeros((d_high,)), cov=torch.eye(d_high), n=n)
    
    #### Sanity Check: 
    #### ensure that SourceX generates d dimensional observations 
    #### and that SourceX2 generates d_high dimensional observations 
    #### if there is a mismatch, then change the d and d_high values to 
    #### match the dimension of the respective observations. 
    x, x2 = SourceX(2), SourceX2(2)
    d_, d_high_ = x.shape[1], x2.shape[1]
    if d_!=d:
        print(f"Input low dimension is {d}, while source generates {d_} dimensional observations")
        print("Setting the low dimension equal to the source dimension")
        d = d_
    if d_high_!=d_high:
        print(f"Input high dimension is {d_high}, while source generates {d_high_} dimensional observations")
        print("Setting the high dimension equal to the source dimension")
        d_high = d_high_
    ####


    CrossMMDVals = torch.zeros((num_trials,))
    MMDVals = torch.zeros((num_trials,))

    CrossMMDVals2 = torch.zeros((num_trials,))
    MMDVals2 = torch.zeros((num_trials,))

    for i in tqdm(range(num_trials)):
        X = SourceX(n)
        Y = SourceX(m)
        bw = get_median_bw(X=X, Y=Y)

        X2, Y2 = SourceX2(n), SourceX2(m)
        bw2 = get_median_bw(X=X2, Y=Y2)

        if kernel_type=='RBF':
            kernel_func = partial(RBFkernel1, bw=bw)
            kernel_func2 = partial(RBFkernel1, bw=bw2)
        elif kernel_type=='Linear':
            kernel_func = LinearKernel
            kernel_func2 = LinearKernel
        elif kernel_type == 'Polynomial':
            kernel_func = partial(PolynomialKernel, scale=bw, degree=poly_degree)
            kernel_func2 = partial(PolynomialKernel, scale=bw2, degree=poly_degree)

        # get the values of the cMMD statistic
        CrossMMDVals[i] = crossMMD2sampleUnpaired(X=X, Y=Y, kernel_func=kernel_func)
        CrossMMDVals2[i] = crossMMD2sampleUnpaired(X=X2, Y=Y2, kernel_func=kernel_func2)

        # get the values of MMD statistic, normalized by their resampled standard deviation
        mmd_func = partial(TwoSampleMMDSquared, unbiased=True, return_float=True)
        MMDVals[i] = mmd_func(X, Y, kernel_func)
        MMDVals2[i] = mmd_func(X2, Y2, kernel_func=kernel_func2)
        std1 = get_resampled_std(X, Y, stat_func=mmd_func, kernel_func=kernel_func)
        std2 = get_resampled_std(X2, Y2, stat_func=mmd_func, kernel_func=kernel_func2)
        MMDVals[i] /= std1
        MMDVals2[i] /= std2

    # Plot the result 
    xx = np.linspace(-10, 10, 1000)
    pp = stats.norm.pdf(xx) # the normal pdf 

    CrossMMDVals = CrossMMDVals.numpy()
    CrossMMDVals2 = CrossMMDVals2.numpy()

    MMDVals = MMDVals.numpy()
    MMDVals2 = MMDVals2.numpy()


    if figname is None:
        figname = '../data/' +  f'Null_Dists_d_{d}_{d_high}_n_{n}_m_{m}_kernel_'
        figname = figname + kernel_type
        timestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        figname = figname + timestr
    crossmmdfigname = figname + 'cross.tex'
    mmdfigname = figname + 'mmd.tex'


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(x=[CrossMMDVals, CrossMMDVals2], density=True, label=[f'd={d}', f'd={d_high}'], alpha=0.8)
    ax.plot(xx, pp, label='N(0,1)', color='k') 
    ax.set_ylabel('Probability density', fontsize=18)
    ax.set_title(f'x-MMD (n={n}, m={m})', fontsize=20)
    ax.legend(fontsize=16)
    if save_fig:
        tpl.save(crossmmdfigname, axis_width=r'\figwidth', axis_height=r'\figheight')
    else:
        plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(x=[MMDVals, MMDVals2], density=True, label=[f'mmd (d={d})', f'mmd (d={d_high})'], alpha=0.8)
    ax.plot(xx, pp, label='N(0,1)', color='k') 
    ax.set_ylabel('Probability density', fontsize=18)
    ax.set_title(f'MMD (n={n}, m={m})', fontsize=20)
    ax.legend(fontsize=16)
    if save_fig:
        tpl.save(mmdfigname, axis_width=r'\figwidth', axis_height=r'\figheight')
    else:
        plt.show()

    if save_data: 
        if filename is None:
            filename = figname + '.pkl'

        results = {}
        results['n'] = n
        results['m'] = m
        results['d'] = d
        results['d_high'] = d_high
        results['num_trials'] = num_trials
        results['cMMD1'] = CrossMMDVals
        results['cMMD2'] = CrossMMDVals2
        results['MMD'] = MMDVals
        results['MMD2'] = MMDVals2

        with open(filename, "wb") as f:
            pickle.dump(results, f)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_low', type=int, help='dimension of the low-dimensional observation space',
                        default=10)
    parser.add_argument('--d_high', type=int, help='dimension of the high-dimensional observation space',
                        default=500)
    parser.add_argument('--use_dirichlet', action='store_true', 
                        help="if true, use the dirichlet distribution")
    parser.add_argument('--n', type=int, default=500, help='Sample Size of X')
    parser.add_argument('--m', type=int, default=500, help='Sample Size of Y')
    parser.add_argument('--num_trials', type=int, default=2000,
                        help='number of trials for getting the null distribution')
    parser.add_argument('--kernel_type', type=str, choices=['RBF', 'Polynomial'], 
                            default='RBF')
    parser.add_argument('--poly_degree', type=int, default=2,
                        help='degree of polynomial kernel')
    parser.add_argument('--save_fig', action='store_true',
                            help="choose whether to save the figures or not")
    parser.add_argument('--save_data', action='store_true',
                            help="choose whether to save the data or not")
    

    args = parser.parse_args()
    d = args.d_low
    d_high = args.d_high
    n, m = args.n, args.m
    num_trials = args.num_trials

    save_data = args.save_data
    save_fig = args.save_fig
    poly_degree= args.poly_degree
    kernel_type = args.kernel_type
    use_dirichlet = args.use_dirichlet

    if use_dirichlet:
        Alpha = 2*np.ones((d,))
        Alpha2 = 2*np.ones((d_high,))
        # generate two dirichlet sources
        def SourceX(n):
            return DirichletVector(d=d, n=n, Alpha=Alpha)
        def SourceX2(n):
            return DirichletVector(d=d_high, n=n, Alpha=Alpha2)
    else:
        # default sources are Gaussian, so no need to specify here.
        SourceX=None 
        SourceX2=None

    figname = '../data/' +  f'Null_Dists_d_{d}_{d_high}_n_{n}_m_{m}_kernel_'
    if use_dirichlet:
        figname += '_Dirichlet_'
    else:
        figname += '_Gaussian_'
    figname_Poly = figname + f'Poly_{poly_degree}_'
    timestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if kernel_type=='RBF':
        figname = figname + 'RBF_'
        figname += timestr
    else:
        figname = figname + f'Poly_{poly_degree}_'
        figname += timestr

    # Call the main function 
    main(n=n, m=m, d=d, d_high=d_high, SourceX=SourceX, 
            SourceX2=SourceX2, save_fig=save_fig, figname=figname,
            num_trials=num_trials, kernel_type=kernel_type, save_data=save_data)

