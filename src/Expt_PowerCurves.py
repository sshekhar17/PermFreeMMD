"""
Compute the Power curves for different MMD-based two-sample tests
"""
import argparse 
from datetime import datetime 
import numpy as np     
from tqdm import tqdm 
from math import sqrt

import matplotlib.pyplot as plt 
plt.style.use('seaborn-white')

from utils import *
from MMDutils import * 
import tikzplotlib as tpl 

                                                                
#########################################################################
def main(SourceX, SourceY, kernel_func=None,  n=200, m=200, num_trials=200,
            num_perms=200, num_bootstrap=200, alpha=0.05, num_points=20,
            block_size_exponent = 0.5, methods=None, kernel_type='RBF',
            poly_degree=2, initial_sample_size=10, save_fig=False,
            save_data=False, figname=None, filename=None, title_info=None, 
            mode=1, num_pts_bw=50):
    
    """
    Compute the power curves of different mmd tests 

    SourceX     : function handle for generate X samples 
    SourceY     : function handle for generate Y samples 
    kernel_func : function hand for the pos-def kernel
    (n, m)      : number of X and Y observations
    num_trials  : number of repetitions for estimating the power 
    num_perms   : number of permutations to be used by permutation test
    alpha       : float denoting the significance level
    methods     : list of string indicating the names of tests to be compared
    num_points  : number of points in the power curves
    kernel_type : "RBF" or "Polynomial" or "Linear"
    poly_degree : int denoting the degree of polynomial, if kernel_type="Polynomial"
    save_fig    : if True, save the figures
    save_data   : if True, save the data 
    figname     : string denoting the name of the figure (used if save_fig==True)
    filename    : string denoting the name of the files (used if save_data==True)
    title_info  : string to be used in generating the title of the figures
    mode        : int in {1, 2}, used in calling the "median_bw_selector" function
    num_pts_bw  : int, used in calling the "median_bw_selector" function
    """
   
    # initialize the sample-sizes to be used in generating the 
    # power curves
    NN = np.linspace(initial_sample_size, n, num_points, dtype=int)
    MM = np.linspace(initial_sample_size, n, num_points, dtype=int)

    # the names of methods whose power curves are to be plotted
    if methods is None: 
        methods = ['mmd-perm', 'c-mmd', 'mmd-spectral', 'b-mmd', 'l-mmd', 'predicted']

    # only implemented block and linear mmd in the paired case
    if 'b-mmd' in methods or 'l-mmd' in methods:
        temp = max(m, n)
        m, n = temp, temp

    #set up function handles for different threshold computing methods
    thresh_permutation = partial(get_bootstrap_threshold, num_perms=num_perms) 
    thresh_normal = get_normal_threshold
    thresh_spectral = partial(get_spectral_threshold,  alpha=alpha, numNullSamp=200)

    # initialize the dictionaries to store the mean and std of the power 
    # of the different tests in the list "method"
    PowerDict = {}
    PowerStdDevDict = {}
    for method in methods:
        PowerDict[method] = np.zeros((num_trials, len(NN)))
        PowerStdDevDict[method] = np.zeros(NN.shape)

    # the main loop 
    for i in tqdm(range(num_trials)):
        for j, (ni, mi) in enumerate(zip(NN, MM)):
            # generate the data for this trial
            X, Y = SourceX(ni), SourceY(mi) 
            # obtain the bandwidth of the kernel
            bw = median_bw_selector(SourceX, SourceY, X, Y, mode, num_pts_bw)
            # initialize the kernel
            if kernel_func is None: # default is to use the RBF kernel
                if kernel_type=='RBF' or kernel_type is None:
                    kernel_type='RBF' # just in case it is None
                    kernel_func = partial(RBFkernel1, bw=bw)
                elif kernel_type=='Linear':
                    kernel_func = LinearKernel 
                elif kernel_type=='Polynomial':
                    if poly_degree is None:
                        kernel_func = partial(PolynomialKernel, degree=2, scale=bw)
                    else:
                        kernel_func = partial(PolynomialKernel, degree=poly_degree,
                        scale=bw)
            # set up function handles for the different statistics 
            unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True) 
            biased_mmd2 = partial(TwoSampleMMDSquared, unbiased=False) 
            cross_mmd2 = crossMMD2sampleUnpaired
            # run all the tests contained in "methods"
            for method in methods:
                if method=='mmd-perm':
                    stat = unbiased_mmd2(X, Y, kernel_func)
                    th = thresh_permutation(X, Y, kernel_func, unbiased_mmd2, alpha=alpha)
                elif method=='mmd-spectral':
                    stat = ni*biased_mmd2(X, Y, kernel_func)
                    th = thresh_spectral(X, Y, kernel_func,  alpha=alpha)
                elif method=='c-mmd':
                    stat = cross_mmd2(X, Y, kernel_func)
                    th = thresh_normal(alpha)
                elif method=='l-mmd':
                    # linear_mmd2 = partial(LinearMMDrbf, bw=bw_)
                    linear_mmd2 = partial(BlockMMDSquared, b=2, return_sig=True, biased=False)
                    stat, sig = linear_mmd2(X, Y, kernel_func)
                    th = sig*thresh_normal(alpha)
                elif method =='b-mmd':
                    block_mmd2 = partial(BlockMMDSquared, b=max(2, int(sqrt(ni))),
                                            return_sig=True, biased=False)
                    stat, sig = block_mmd2(X, Y, kernel_func)
                    th = sig*thresh_normal(alpha)
                # record the outcome of this test
                PowerDict[method][i][j] = 1.0*(stat>th)
    #compute the mean and std of the power of all methods
    for method in methods:
        PowerStdDevDict[method] = np.array([
            get_bootstrap_std(PowerDict[method][:, i], num_bootstrap=num_bootstrap) 
            for i in range(len(NN)) ])
        PowerDict[method] = PowerDict[method].mean(axis=0) 
    # predict the power of the permutation-kernel-MMD test 
    # using the observed power of cross-MMD test 
    # see Eq. (9) of Shekar, Kim, and Ramdas, Neurips 2022. 
    if 'predicted' in methods:
        pred_power = predict_power(PowerDict['c-mmd'],alpha=alpha)
        PowerDict['predicted'] = pred_power
        # heuristic calculation of the uncertainty: sqrt(p*(1-p)/n)
        PowerStdDevDict['predicted'] = np.sqrt(pred_power*(1-pred_power)/num_trials)

    # Generate the results dict 
    palette = sns.color_palette(palette='tab10', n_colors=10)
    Results = {}
    Results['num_trials'] = num_trials 
    Results['n'] = n 
    Results['epsilon'] = epsilon
    Results['block_size_exponent'] = block_size_exponent
    Results['SourceX'] = SourceX
    Results['SourceY'] = SourceY
    Results['num_perms'] = num_perms
    Results['PowerDict'] = PowerDict
    Results['PowerStdDevDict'] = PowerStdDevDict
    Results['methods'] = methods
    Results['palette'] = palette
    # plot the results
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    for i, method in enumerate(methods):
        # get the mean and standard deviation of power for this method
        pm, ps = PowerDict[method], PowerStdDevDict[method]
        if method=='predicted': # use dashed lines for predicted power
            ax.plot(NN+MM, pm, '--', color=palette[i], label=method)
            ax.fill_between(NN+MM, pm-ps, pm+ps, color=palette[i], alpha=0.3)
        else:
            ax.plot(NN+MM, pm, color=palette[i], label=method)
            ax.fill_between(NN+MM, pm-ps, pm+ps, color=palette[i], alpha=0.3)
    if title_info is not None:
        ax.set_title(f"Power vs Sample-Size " +title_info, fontsize=16)
    else:
        ax.set_title(f"Power vs Sample-Size", fontsize=16)
    ax.set_ylabel('Power', fontsize=14)

    ax.set_xlabel('Sample-Size (n+m)', fontsize=14)
    if len(methods)>3:
        # ax.legend(bbox_to_anchor=(1.01, 1.02), fontsize=14, frameon=True)
        ax.legend(fontsize=14, frameon=True)
    else:
        ax.legend(fontsize=14)
    # obtain the name of the figure 
    if figname is None:
        figname = 'PowerCurve_'+kernel_type
        timestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        figname = '../data/' + figname + timestr + '_.tex'
        print(figname)
    if save_fig:
        # plt.savefig(figname, bbox_inches="tight", dpi=450)
        tpl.save(figname, axis_width=r'\figwidth', axis_height=r'\figheight')
    else:
        plt.show()
    # obtain the name of the file for saving 
    if filename is None:
        filename = 'PowerCurve_'
        timestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = '../data/' + filename + timestr + '_.pkl'
        # print(filename)
    # store the data if required 
    if save_data:
        with open(filename, 'wb') as f:
            pickle.dump(Results, f)
    # store the filename and figname in the results dict 
    Results['figname'] = figname
    Results['filename'] = filename
    return Results

##----
def mainTime(SourceX, SourceY, kernel_func=None,  n=200, m=200, num_trials=200,
            num_perms=200, alpha=0.05, num_points=20,
            block_size_exponent = 0.5, methods=None, kernel_type='RBF',
            poly_degree=2, initial_sample_size=10, save_fig=False,
            figname=None,  title_info=None, mode=1, num_pts_bw=50):
    """
        Plot the running-time vs power for different tests 
        SourceX     : function handle for generate X samples 
        SourceY     : function handle for generate Y samples 
        kernel_func : function hand for the pos-def kernel
        (n, m)      : number of X and Y observations
        num_trials  : number of repetitions for estimating the power 
        num_perms   : number of permutations to be used by permutation test
        alpha       : float denoting the significance level
        methods     : list of string indicating the names of tests to be compared
        num_points  : number of points in the power curves
        kernel_type : "RBF" or "Polynomial" or "Linear"
        poly_degree : int denoting the degree of polynomial, if kernel_type="Polynomial"
        save_fig    : if True, save the figures
        save_data   : if True, save the data 
        figname     : string denoting the name of the figure (used if save_fig==True)
        filename    : string denoting the name of the files (used if save_data==True)
        title_info  : string to be used in generating the title of the figures
        mode        : int in {1, 2}, used in calling the "median_bw_selector" function
        num_pts_bw  : int, used in calling the "median_bw_selector" function

    """
    if methods is None: 
        methods = ['mmd-perm', 'c-mmd']

    # only implemented block and linear mmd in the paired case
    if 'b-mmd' in methods or 'l-mmd' in methods:
        temp = max(m, n)
        m, n = temp, temp
    
    # initialize the set of sample-sizes to be used 
    NN = np.linspace(initial_sample_size, n, num_points, dtype=int)
    MM = np.linspace(initial_sample_size, m, num_points, dtype=int)

    #set up function handles for different threshold computing methods
    thresh_permutation = partial(get_bootstrap_threshold, num_perms=num_perms) 
    thresh_normal = get_normal_threshold
    thresh_spectral = partial(get_spectral_threshold,  alpha=alpha, numNullSamp=200)

    # initialize the dictionary for storing the running times and 
    # powers of different tests 
    TimesDict, PowerDict =  {}, {}
    for method in methods:
        assert NN.shape == MM.shape 
        PowerDict[method] = np.zeros(NN.shape)
        TimesDict[method] = np.zeros(NN.shape)

    # the main loop 
    for i in tqdm(range(num_trials)):
        for j, (ni, mi) in enumerate(zip(NN, MM)):
            # generate the data for the current trial
            X, Y = SourceX(ni), SourceY(mi) 
            # get the bandwidth, and the kernel
            bw = median_bw_selector(SourceX, SourceY, X, Y, mode, num_pts_bw)
            if kernel_func is None: # default is to use the RBF kernel
                if kernel_type=='RBF' or kernel_type is None:
                    kernel_type='RBF' # just in case it is None
                    kernel_func = partial(RBFkernel1, bw=bw)
                elif kernel_type=='Linear':
                    kernel_func = LinearKernel 
                elif kernel_type=='Polynomial':
                    if poly_degree is None:
                        kernel_func = partial(PolynomialKernel, degree=2, scale=bw)
                    else:
                        kernel_func = partial(PolynomialKernel, degree=poly_degree,
                        scale=bw)
            # set up function handles for the different statistics 
            unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True) 
            biased_mmd2 = partial(TwoSampleMMDSquared, unbiased=False) 
            cross_mmd2 = crossMMD2sampleUnpaired
            # run the differnet tests
            for method in methods:
                start_time = time()
                if method=='mmd-perm':
                    stat = unbiased_mmd2(X, Y, kernel_func)
                    th = thresh_permutation(X, Y, kernel_func, unbiased_mmd2, alpha=alpha)
                elif method=='mmd-spectral':
                    stat = ni*biased_mmd2(X, Y, kernel_func)
                    th = thresh_spectral(X, Y, kernel_func,  alpha=alpha)
                elif method=='c-mmd':
                    stat = cross_mmd2(X, Y, kernel_func)
                    th = thresh_normal(alpha)
                elif method=='l-mmd':
                    # linear_mmd2 = partial(LinearMMDrbf, bw=bw_)
                    linear_mmd2 = partial(BlockMMDSquared, b=2, return_sig=True, biased=False)
                    stat, sig = linear_mmd2(X, Y, kernel_func)
                    th = sig*thresh_normal(alpha)
                elif method =='b-mmd':
                    block_mmd2 = partial(BlockMMDSquared, b=max(2, int(sqrt(ni))),
                                            return_sig=True, biased=False)
                    stat, sig = block_mmd2(X, Y, kernel_func)
                    th = sig*thresh_normal(alpha)
                # record the outcome of this test
                running_time = time() - start_time 
                TimesDict[method][j] += running_time 
                PowerDict[method][j] += 1.0*(stat>th)
    # obtain the power and average running times
    for method in methods:
        PowerDict[method] /=  num_trials
        TimesDict[method] /=  num_trials

    # Generate the results dict 
    palette = sns.color_palette(palette='tab10', n_colors=10)
    Results = {}
    Results['num_trials'] = num_trials 
    Results['n'] = n 
    Results['epsilon'] = epsilon
    Results['block_size_exponent'] = block_size_exponent
    Results['SourceX'] = SourceX
    Results['SourceY'] = SourceY
    Results['num_perms'] = num_perms
    Results['TimesDict'] = TimesDict
    Results['PowerDict'] = PowerDict
    Results['methods'] = methods
    Results['palette'] = palette

    # plot the figures 
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    markers = ['o', '^', 's', 'p', '*', 'P']
    for i, method in enumerate(methods):
        # get the mean and standard deviation of power for this method
        times, power = TimesDict[method], PowerDict[method] 
        sizes = 10 + power * 60 # maximum size is 20  
        ax.scatter(times, power, s=sizes, label=method, color=palette[i], 
                        marker=markers[i], alpha=0.7, edgecolors='k')
    if title_info is not None:
        ax.set_title(f"Power vs Running time" +title_info, fontsize=16)
    else:
        ax.set_title(f"Power vs Running time", fontsize=16)
    ax.set_ylabel('Power', fontsize=14)

    ax.set_xlabel('Running time (seconds)', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=14)
    # obtain the name of the figure 
    if figname is None:
        figname = 'PowerVsTimes_'+kernel_type
        timestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        figname = '../data/' + figname + timestr + '_.tex'
        print(figname)
    if save_fig:
        tpl.save(figname, axis_width=r'\figwidth', axis_height=r'\figheight')
    else:
        plt.show()
    # store the figname 
    Results['figname'] = figname
    return Results


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, help='dimension of observation space',
                        default=10)
    parser.add_argument('--eps', type=float, help='magnitude of perturbation',
                        default=0.2)
    parser.add_argument('--num_pert', type=int, default=5, 
                        help='number of coordinates out of d to perturb under alt')
    parser.add_argument('--n', type=int, default=100, help='Sample Size of X')
    parser.add_argument('--m', type=int, default=100, help='Sample Size of Y')
    parser.add_argument('--alpha', type=float, default=0.05, help='level of test')
    parser.add_argument('--initial', type=int, default=20, help="minimum sample size")
    parser.add_argument('--num_points', type=int, default=20,
                        help='number of sample-sizes between args.initial and n,m to be used')
    parser.add_argument('--num_perms', type=int, default=50,
                        help='number of permutations to be used')
    parser.add_argument('--num_trials', type=int, default=50,
                        help='number of trials for estimating power')
    parser.add_argument('--kernel_type', type=str, choices=['RBF', 'Polynomial'], 
                        default='RBF')
    parser.add_argument('--poly_degree', type=int, default=2,
                        help='degree of polynomial kernel')
    parser.add_argument('--bmmd_exp', type=float, default=0.5, 
                        help='exponent of the block size')
    parser.add_argument('--save_fig', action='store_true',
                            help="choose whether to save the figures or not")
    parser.add_argument('--save_data', action='store_true',
                            help="choose whether to save the data or not")
    parser.add_argument('--time_expt', action='store_true',
                            help="if true, run the experiment comparing \\\
                            running times of cMMD and MMD")
    parser.add_argument('--mode', choices={1, 2}, default=1,
                            help="mode for selecting bandwidth via median heuristic")
    parser.add_argument('--num_pts_bw', type=int, default=25,
                            help="number of data-points for median heuristic")

    args = parser.parse_args()
    d = args.d
    epsilon = args.eps
    n,m = args.n, args.m
    alpha = args.alpha
    initial_sample_size = args.initial
    num_points = args.num_points
    save_fig = args.save_fig
    save_data = args.save_data
    num_perturbations = args.num_pert
    num_perms = args.num_perms 
    num_trials = args.num_trials
    kernel_type=args.kernel_type 
    poly_degree = args.poly_degree
    block_size_exponent=args.bmmd_exp 
    time_expt = args.time_expt 
    mode = args.mode 
    num_pts_bw = args.num_pts_bw 
    

    print(d, epsilon, n, m, alpha, num_points, save_fig, kernel_type)

    # create the data sources 
    meanX, meanY = np.ones((d,)),  np.ones((d,))
    covX, covY = np.eye(d), np.eye(d)
    meanY[:num_perturbations] = (1+epsilon)
    def SourceX(n):
        return GaussianVector(mean=meanX, cov=covX, n=n)

    def SourceY(n):
        return GaussianVector(mean=meanY, cov=covY, n=n)

    # information used in the title of the figures        
    title_info = f"(d={d}, j={num_perturbations}, $\epsilon$={epsilon})"
    # title_info = f"(d={d}, $\epsilon$={epsilon})"
    if time_expt:     
        methods = ['mmd-perm', 'c-mmd']
        temp_ = '../data/PowerVsTime_'+kernel_type + f'd_{d}_eps_{epsilon}'.replace('.', '_')
        figname = temp_ + '.tex'
        Results = mainTime(SourceX, SourceY, kernel_func=None,  n=n, m=n,
                        num_trials=num_trials, num_perms=num_perms,
                        alpha=alpha, num_points=num_points,
                        block_size_exponent = 0.5, methods=methods,
                        initial_sample_size=initial_sample_size, save_fig=save_fig, 
                        figname=figname, title_info=title_info, kernel_type=kernel_type, 
                        poly_degree=poly_degree, mode=mode, num_pts_bw=num_pts_bw)
        
    else:
        methods = ['mmd-perm', 'c-mmd',  'predicted', 'mmd-spectral', 'b-mmd', 'l-mmd']
        # methods = ['mmd-perm', 'c-mmd', 'predicted']
        temp_ = '../data/PowerCurveAllMethods_'+kernel_type + f'd_{d}_eps_{epsilon}'.replace('.', '_')
        figname = temp_ + '.tex'
        filename = temp_ + '.pkl'
        Results = main(SourceX, SourceY, kernel_func=None,  n=n, m=n,
                        num_trials=num_trials, num_perms=num_perms,
                        alpha=alpha, num_points=num_points,
                        block_size_exponent = 0.5, methods=methods,
                        initial_sample_size=initial_sample_size, save_fig=save_fig, 
                        save_data=save_data, figname=figname, filename=filename, 
                        title_info=title_info, kernel_type=kernel_type, 
                        poly_degree=poly_degree, mode=mode, num_pts_bw=num_pts_bw)
        
