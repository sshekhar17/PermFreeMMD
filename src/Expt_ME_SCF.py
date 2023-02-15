import argparse
from time import time 
import numpy as np
import matplotlib.pyplot as plt
import freqopttest.util as util
import freqopttest.data as data
import freqopttest.kernel as kernel
import freqopttest.tst as tst
import freqopttest.glo as glo
from tqdm import tqdm 

import tikzplotlib as tpl 

from MMDutils import runCMMDexperiment
from math import log

##TODO: fix SCF warnings at smaller sample-size values 
#_______________________________________________________________________________
# Solution to hide the output stream of ME and SCF tests
# This snippet of code is copied from Alexander C's 
# answer at this link: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print#:~:text=If%20you%20don't%20want,the%20top%20of%20the%20file. 
import os, sys

from utils import GaussianVector

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
#_______________________________________________________________________________


def ME_test(NN, dim, seed, my, num_trials, alpha=0.05, J=5, progress_bar=True):
    if seed is None:
        seed = int(time())
    Power = np.zeros(NN.shape)

    range_ = enumerate(tqdm(NN)) if progress_bar else enumerate(NN)
    for i, n in range_:

        for trial in range(num_trials):
            ss = data.SSGaussMeanDiff(dim, my=my)
            # draw n points from P, and n points from Q. 
            # tst_data is an instance of TSTData
            tst_data = ss.sample(n, seed=seed + i*1000 + trial*10)
            tr, te = tst_data.split_tr_te(tr_proportion=0.5, seed=np.random.randint(10000))

            # These are options to the optimizer. Many have default values. 
            # See the code for the full list of options.
            op = {
                'n_test_locs': J, # number of test locations to optimize
                'max_iter': 200, # maximum number of gradient ascent iterations
                'locs_step_size': 1.0, # step size for the test locations (features)
                'gwidth_step_size': 0.1, # step size for the Gaussian width
                'tol_fun': 1e-4, # stop if the objective does not increase more than this.
                'seed': seed+5,  # random seed
            }

            with HiddenPrints():
                # optimize on the training set
                test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(tr, alpha, **op)
            # Construct a MeanEmbeddingTest object with the best optimized test features, 
            # and optimized Gaussian width
            met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
            # Do the two-sample test on the test data te. 
            # The returned test_result is a dictionary.
            test_result = met_opt.perform_test(te)
            if test_result['h0_rejected']:
                Power[i] += 1.0

    Power /= num_trials
    return Power

def SCF_test(NN, dim, seed, my, num_trials, alpha=0.05, J=1, progress_bar=True):
    
    Power = np.zeros(NN.shape)

    range_ = enumerate(tqdm(NN)) if progress_bar else enumerate(NN)
    for i, n in range_:

        for trial in range(num_trials):
            ss = data.SSGaussMeanDiff(dim, my=my)
            # draw n points from P, and n points from Q. 
            # tst_data is an instance of TSTData
            tst_data = ss.sample(n, seed=seed + i*1000 + trial*10)
            tr, te = tst_data.split_tr_te(tr_proportion=0.5, seed=np.random.randint(10000))

            # These are options to the optimizer. Many have default values. 
            # See the code for the full list of options.
            op = {'n_test_freqs': J, 'max_iter': 200, 'freqs_step_size': 0.1, 
                'gwidth_step_size': 0.1, 'seed': seed+92856, 'tol_fun': 1e-3}
            with HiddenPrints():
                # optimize on the training set
                test_freqs, gwidth, _ = tst.SmoothCFTest.optimize_freqs_width(tr, alpha, **op)
                # Construct a SCFtest object with the best optimized test features, 
                # and optimized Gaussian width
                scf_opt = tst.SmoothCFTest(test_freqs, gwidth, alpha)
                # Do the two-sample test on the test data te. 
                # The returned test_result is a dictionary.
                test_result = scf_opt.perform_test(te)
            if test_result['h0_rejected']:
                Power[i] += 1.0

    Power /= num_trials
    return Power




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dims', default=10, type=int,  help='dimension of features')
    parser.add_argument('-n', '--num_samples', default=200, type=int,  help='sample-size')
    parser.add_argument('-ns', '--num_steps', default=10, type=int,  help='number of steps for plotting power curves')
    parser.add_argument('-a', '--alpha', default=0.05, type=float,  help='significance level of the test')
    parser.add_argument('-nt', '--num_trials', default=200, type=int,  help='number of repetitions to plot power curves')
    parser.add_argument('-e', '--epsilon', default=1.0,  type=float, help='perturbation for the GMD source')
    parser.add_argument('--save_fig', action='store_true',
                            help="choose whether to save the figures or not")
    args = parser.parse_args() 


    dim = args.dims
    epsilon = args.epsilon 
    save_fig=args.save_fig
    n = args.num_samples
    num_steps = args.num_steps
    num_trials = args.num_trials
    alpha = args.alpha

    # different values of sample size for plotting the power curve 
    initial_n = min(20, n) # smallest sample size 
    NN = np.linspace(initial_n, n, num_steps, dtype=int)
    # initialize the random seed 
    seed = int(time())%9999

    ## run the SCF test
    PowerSCF = SCF_test(NN=NN, dim=dim, seed=seed, my=epsilon, alpha=alpha,
                        num_trials=num_trials, J=5)


    # run the ME Test 
    PowerME = ME_test(NN=NN, dim=dim, seed=seed, my=epsilon, alpha=alpha,
                        num_trials=num_trials, J=5)

    # Run the cross-MMD test 
    # PowerDict, TimesDict = main_new(d=dim, epsilon=epsilon, n=n, 
    #                                 num_trials=num_trials,
    #                                 alpha=alpha,
    #                                 num_perturbations=1, # same as the GMD-source 
    #                                 return_data=True, 
    #                                 plot_figs=False,
    #                                 num_steps=num_steps,
    #                                 seed=seed,
    #                                 initial_value=initial_n)

    def SourceX(n_): 
        return GaussianVector(mean=np.zeros((dim,)), cov=np.eye(dim), n=n_) 
    
    def SourceY(m_):
        mean2 = np.zeros((dim,)) 
        mean2[0] = epsilon
        return GaussianVector(mean=mean2, cov=np.eye(dim), n=m_) 

    NN_, MM_, PowerCMMD = runCMMDexperiment(SourceX, SourceY, n, m=n, 
                                kernel_func=None, num_trials=num_trials, alpha=alpha,
                                num_steps=num_steps, initial_value=initial_n)
    print(f'Power of SCF Test')
    print(PowerSCF)
    print('\n')

    print(f'Power of ME Test')
    print(PowerME)
    print('\n')

    print("Power of C-MMD Test")
    print(PowerCMMD)

    # (n+m)
    NN = NN * 2

    plt.plot(NN, PowerCMMD, label='x-MMD')
    plt.plot(NN, PowerME, label='ME')
    plt.plot(NN, PowerSCF, label='SCF')
    plt.xlabel('Sample-Size (n+m)', fontsize=13)
    if epsilon==0:
        plt.ylabel('Type-I error', fontsize=13)
        plt.plot(NN, alpha*np.ones(NN.shape), 'k--', alpha=0.5)
        plt.title(f'GMD Source: (d={dim}, {num_trials} trials)', fontsize=15)
    else:
        plt.ylabel('Power', fontsize=13)
        plt.title(f'GMD Source: ($\epsilon$= {epsilon}, d={dim})', fontsize=15)
    plt.legend(fontsize=12)

    # plt.show()
    if epsilon==0:
        figname = '../data/' + f'Type_I_comparison_'+f'd_{dim}_seed_{seed}_.tex'
    else:
        figname ='../data/' + f'Power_comparison_my_{epsilon}'.replace('.', '_') +f'd_{dim}_seed_{seed}_.tex'
    if save_fig:
        tpl.save(figname, axis_width=r'\figwidth', axis_height=r'\figheight')
    else:
        plt.show()

