import argparse 
from tqdm import tqdm
from functools import partial 
import torch
from math import sqrt 
import numpy as np 
import matplotlib.pyplot as plt
from MMDutils import get_median_bw 
plt.style.use('seaborn-white')

from utils import PolynomialKernel, RBFkernel1 
from utils import TwoSampleMMDSquared, BlockMMDSquared, crossMMD2sampleUnpaired
from utils import DirichletVector, GaussianVector
import tikzplotlib as tpl 

def main(n, num_null, SourceX=None, SourceY=None, save_fig=False, figname=None,
            figtitle=None, kernel_type='RBF', poly_degree=5):
    # same number of alt and null data-points 
    num_alt=num_null


    # set up function handles for the different statistics 
    unbiased_mmd = partial(TwoSampleMMDSquared, unbiased=True) 
    linear_mmd = partial(BlockMMDSquared, b=2, biased=False)
    block_mmd = partial(BlockMMDSquared, b=int(sqrt(n)), biased=False)
    block_mmd2 = partial(BlockMMDSquared, b=int(n**0.33), biased=False)
    cross_mmd = crossMMD2sampleUnpaired 

    # default null and alt sources 
    if SourceX is None or SourceY is None:
        d = 10
        epsilon = 0.3
        num_perturbations = 5

        meanX = torch.ones((d,))
        meanY = torch.ones((d,))
        meanY[:num_perturbations] = 1+epsilon
        covX = torch.eye(d)
        covY = torch.eye(d)

        def SourceX(n):
                return GaussianVector(mean=meanX, cov=covX, n=n)

        def SourceY(n):
                return GaussianVector(mean=meanY, cov=covY, n=n)

    # compute the statistics 
    N = num_null + num_alt
    mmd_vals = torch.zeros((N,))
    linear_mmd_vals = torch.zeros((N,))
    block_mmd_vals1 = torch.zeros((N,))
    block_mmd_vals2 = torch.zeros((N,))
    cmmd_vals = torch.zeros((N,))

    for i in tqdm(range(num_null)):
        # draw the null sample 
        Xn = SourceX(n)
        Yn = SourceX(n) 

        bw = get_median_bw(X=Xn, Y=Yn)
        if kernel_type=='RBF':
            kernel_func = partial(RBFkernel1, bw=bw)
        elif kernel_type=='Polynomial':
            if poly_degree is None: 
                poly_degree=1 # default is linear kernel
            kernel_func = partial(PolynomialKernel, scale=bw, degree=poly_degree)

        mmd_vals[i] = unbiased_mmd(Xn, Yn, kernel_func, return_float=True)
        linear_mmd_vals[i] = linear_mmd(Xn, Yn, kernel_func)
        block_mmd_vals1[i] = block_mmd(Xn, Yn, kernel_func)
        block_mmd_vals2[i] = block_mmd2(Xn, Yn, kernel_func)
        cmmd_vals[i] = cross_mmd(Xn, Yn, kernel_func)

        # draw the alt samples 
        Xa, Ya = SourceX(n), SourceY(n) 

        bw1 = get_median_bw(X=Xn, Y=Yn)
        if kernel_type=='RBF':
            kernel_func1 = partial(RBFkernel1, bw=bw1)
        elif kernel_type=='Polynomial':
            if poly_degree is None: 
                poly_degree=1 # default is linear kernel
            kernel_func1 = partial(PolynomialKernel, scale=bw1, degree=poly_degree)


        mmd_vals[i+num_null] = unbiased_mmd(Xa, Ya, kernel_func1, return_float=True)
        linear_mmd_vals[i+num_null] = linear_mmd(Xa, Ya, kernel_func1)
        block_mmd_vals1[i+num_null] = block_mmd(Xa, Ya, kernel_func1)
        block_mmd_vals2[i+num_null] = block_mmd2(Xa, Ya, kernel_func1)
        cmmd_vals[i+num_null] = cross_mmd(Xa, Ya, kernel_func1)

    # change to numpy
    mmd_vals = mmd_vals.numpy()
    linear_mmd_vals = linear_mmd_vals.numpy()
    block_mmd_vals1 = block_mmd_vals1.numpy()
    block_mmd_vals2 = block_mmd_vals2.numpy()
    cmmd_vals = cmmd_vals.numpy()

    # sorted statistics (for selecting the thresholds)
    mmd_vals_ = np.sort(mmd_vals)
    linear_mmd_vals_ = np.sort(linear_mmd_vals)
    cmmd_vals_ = np.sort(cmmd_vals)
    block_mmd_vals1_= np.sort(block_mmd_vals1)
    block_mmd_vals2_= np.sort(block_mmd_vals2)


    FPmmd, TPmmd = np.zeros((N,)), np.zeros((N,))
    FPlmmd, TPlmmd = np.zeros((N,)), np.zeros((N,))
    FPcmmd, TPcmmd = np.zeros((N,)), np.zeros((N,))
    FPbmmd1, TPbmmd1 = np.zeros((N,)), np.zeros((N,))
    FPbmmd2, TPbmmd2 = np.zeros((N,)), np.zeros((N,))


    def FPTP(j, stats, sorted_stats, num_null):
        th = sorted_stats[j]
        fp = sum(stats[:num_null]>=th)/num_null
        tp = sum(stats[num_null:]>=th)/num_null
        return fp, tp

    for i in range(N):
        #mmd 
        FPmmd[i], TPmmd[i] = FPTP(i, mmd_vals, mmd_vals_, num_null)
        FPcmmd[i], TPcmmd[i] = FPTP(i, cmmd_vals, cmmd_vals_, num_null)
        FPlmmd[i], TPlmmd[i] = FPTP(i, linear_mmd_vals, linear_mmd_vals_, num_null)
        FPbmmd1[i], TPbmmd1[i] = FPTP(i, block_mmd_vals1, block_mmd_vals1_, num_null)
        FPbmmd2[i], TPbmmd2[i] = FPTP(i, block_mmd_vals2, block_mmd_vals2_, num_null)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    xx = np.linspace(0, 1, 500)
    ax.plot(FPmmd, TPmmd, label='mmd')
    ax.plot(FPcmmd, TPcmmd, label='c-mmd')
    ax.plot(FPbmmd1, TPbmmd1, label='b-mmd $(n^{1/2})$')
    ax.plot(FPbmmd2, TPbmmd2, label='b-mmd $(n^{1/3})$')
    ax.plot(FPlmmd, TPlmmd, label='l-mmd')
    ax.plot(xx, xx, '--', alpha=0.3)
    if figtitle is None:
        figtitle = 'ROC curves'
    ax.set_title(figtitle, fontsize=18)
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)

    # ax.legend(bbox_to_anchor=(0.5, 0.02), fontsize=14, frameon=True)
    ax.legend(loc = "lower right", fontsize=14, frameon=True)
    if save_fig:
        if figname is None:
            figname = f'../data/ROC_curve_d_{d}_n_{n}_N_{N}_final.tex'
        # plt.savefig(figname, bbox_inches="tight", dpi=450)
        tpl.save(figname, axis_width=r'\figwidth', axis_height=r'\figheight')
    else:
        plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, help='dimension of the observation space',
                        default=10)
    parser.add_argument('--n', type=int, default=200, help='Sample Size')
    parser.add_argument('--eps', type=float, default=0.25, help='magnitude of perturbation')
    parser.add_argument('--num_pert', type=int, default=5, 
                        help='number of coordinates out of d to perturb under alt')
    parser.add_argument('--use_dirichlet', action='store_true', 
                        help="if true, use the dirichlet distribution")
    parser.add_argument('--num_trials', type=int, default=1000,
                        help='number of trials for getting the null distribution')
    parser.add_argument('--kernel_type', type=str, choices=['RBF', 'Polynomial'])
    parser.add_argument('--poly_degree', type=int, default=2,
                        help='degree of polynomial kernel')
    parser.add_argument('--save_fig', action='store_true',
                            help="choose whether to save the figures or not")
   

    args = parser.parse_args()
    d = args.d
    n = args.n #we use n=m in this experiment
    save_fig = args.save_fig
    num_null=args.num_trials
    epsilon=args.eps
    num_perturbations = args.num_pert
    use_dirichlet = args.use_dirichlet

    if use_dirichlet:
        AlphaX = 1*np.ones((d,)) 
        AlphaY = (1+epsilon)*np.ones((d,))
        def SourceX(n):
            return DirichletVector(d=d, n=n, Alpha=AlphaX)
        def SourceY(n):
            return DirichletVector(d=d, n=n, Alpha=AlphaY)

        figname= '../data/'
        figname += f'ROC_curve_n_{n}_d_{d}_Dirichlet.tex'
        figtitle = f'Dirichlet (d={d}, $\epsilon$={epsilon})'

    else:
        meanX = torch.ones((d,))
        meanY = torch.ones((d,))
        meanY[:num_perturbations] = 1+epsilon
        def SourceX(n):
            return GaussianVector(mean=meanX, cov=torch.eye(d), n=n)

        def SourceY(n):
            return GaussianVector(mean=meanY, cov=torch.eye(d), n=n)

        figname= '../data/'
        figname += f'ROC_curve_n_{n}_d_{d}_eps_{epsilon}.tex'
        figtitle = f'd={d}, $\epsilon$={epsilon}, j={num_perturbations}, n={n}'


    main(n, num_null, SourceX, SourceY, save_fig=save_fig, figname=figname, 
            figtitle=figtitle)