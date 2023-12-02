import torch 
from math import sqrt, log 
from functools import partial 
import numpy as np 
import scipy.stats as stats 
from scipy.spatial.distance import cdist, pdist 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
plt.style.use('seaborn-white')

###====================Kernel Functions=======================
def RBFkernel(x, y=None, bw=1.0, amp=1.0):
    """
        k(x, y) = amp * exp( - \|x-y\|^2 / (2*bw^2)) 
    """
    y = x if y is None else y
    dists = cdist(x, y)
    squared_dists = dists * dists 
    k = amp * np.exp( -(1/(2*bw*bw)) * squared_dists ) 
    return k 


def LinearKernel(x, y=None):
    """
        k(x, y) = x^T y
    """
    y = x if y is None else y 
    k = np.einsum('ji, ki -> jk', x, y)
    return k 


def PolynomialKernel(x, y=None, c=1.0, scale=1.0,  degree=2):
    """
        k(x, y) = ( c + (x^T y)/scale )**degree
    """
    y = x if y is None else y
    # get the matrix of dot-products 
    D = LinearKernel(x=x, y=y)
    # compute the polynomial kernel 
    assert scale!=0
    k = ((D + c)/scale)**degree 
    return k 


def RBFkernel1(x, y=None, bw=1.0, amp=1.0, pairwise=False):
    """
        torch version of the RBF kernel
        k(x, y) = amp * exp( - \|x-y\|^2 / (2*bw*bw)  ) 
    """
    y = x if y is None else y 
    if pairwise:
        assert y.shape==x.shape 
        squared_dists = ((y-x)**2).sum(axis=1)
    else:
        dists = cdist(x, y)
        squared_dists = dists * dists 
    k = amp * np.exp( -(1/(2*bw*bw)) * squared_dists ) 
    return k 


###====================MMD Statistics=======================
def crossMMD2sampleUnpaired(X, Y, kernel_func):
    """
        Compute the studentized cross-MMD statistic
        Details in Section 2 of https://arxiv.org/pdf/2211.14908.pdf 
    """
    n, d = X.shape 
    m, d_ = Y.shape 
    # sanity check 
    assert (d_==d) and (n>=2) and (m>=2) 
    # split the dataset into two equal parts
    n1, m1 = n//2, m//2 
    X1, X2 = X[:n1], X[n1:]
    Y1, Y2 = Y[:m1], Y[m1:]
    # comptue the gram matrices
    Kxx = kernel_func(X1, X2) 
    Kyy = kernel_func(Y1, Y2)
    Kxy = kernel_func(X1, Y2) 
    Kyx = kernel_func(Y1, X2)
    # compute the numerator of the statistic
    # Equation (2) in https://arxiv.org/pdf/2211.14908.pdf
    Ux = Kxx.mean() - Kxy.mean()
    Uy = Kyx.mean() - Kyy.mean()  
    U = Ux - Uy
    # compute the denominator 
    # Equation (4) in https://arxiv.org/pdf/2211.14908.pdf
    term1 = (Kxx.mean(axis=1) - Kxy.mean(axis=1) - Ux)**2
    sigX2 = term1.mean() 
    term2 = (Kyx.mean(axis=1) - Kyy.mean(axis=1) - Uy)**2
    sigY2 = term2.mean() 
    sig = sqrt(sigX2/n1 + sigY2/m1)  
    if not sig>0:
        print(f'term1={term1}, term2={term2}, sigX2={sigX2}, sigY2={sigY2}')
        raise Exception(f'The denominator is {sig}')
    # obtain the statistic
    T = U/sig 
    return T


def TwoSampleMMDSquared(X, Y, kernel_func, unbiased=False, 
                        return_float=False):
    Kxx = kernel_func(X, X) 
    Kyy = kernel_func(Y, Y)
    Kxy = kernel_func(X, Y)

    n, m = len(X), len(Y)

    term1 = Kxx.sum()
    term2 = Kyy.sum()
    term3 = 2*Kxy.mean()

    if unbiased:
        term1 -= np.trace(Kxx)
        term2 -= np.trace(Kyy)
        MMD_squared = (term1/(n*(n-1)) + term2/(m*(m-1)) - term3)
    else:
        MMD_squared = term1/(n*n) + term2/(m*m) - term3 
    if return_float:
        return MMD_squared
    else:
        return MMD_squared 





def LinearMMDrbf(X, Y, kernel_func=None, perm=None, biased=True, bw=None,
                    return_scaled=False):
    """
        This vectorized implementation is taken from the `rbf_mmd2_streaming' 
        function implemented at the link below: 
        https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py`
    """
    
    if not biased:
        print('Not implemented unbiased version: switching to biased')   
    n = (X.shape[0] // 2) * 2
    if bw is None:
        d_ = X.shape[1] 
        bw = sqrt(d_) # a heuristic choice of bw 
    gamma = 1 / (2 * bw**2)
    rbf = lambda A, B: np.exp(-gamma * ((A - B) ** 2).sum(axis=1))
    mmd2_ = (rbf(X[:n:2], X[1:n:2]) + rbf(Y[:n:2], Y[1:n:2])
          - rbf(X[:n:2], Y[1:n:2]) - rbf(X[1:n:2], Y[:n:2]))/2 
    stat = mmd2_.mean()
    if return_scaled:
        sig = mmd2_.std()
        stat = sqrt(n//2)*stat/sig 
    return stat


# def BlockMMDSquaredNew(X, Y, kernel_func, b=2, perm=None, biased=False):
#     """
#         Compute the block mmd squared statistic 
#         This is the python translation of the matlab code written by 
#         W. Zaremba at this link:
#         https://github.com/wojzaremba/btest/blob/master/btest.m
#     """
#     # sanity checks 
#     n, m = len(X), len(Y) 
#     if m>n:
#         Y=Y[:n] 
#         # m=n # we don't use m later in the function
#     elif m<n:
#         n=m 
#         X=X[:n]
#     r = n%b 
#     if r!=0: # drop the last r terms in X and Y
#         n = int(n-r)
#         X, Y = X[:n], Y[:n] 
#     # now compute the statistic 
#     num_blocks = n//b 
#     if perm is None:
#         perm = np.arange(n) 
#     X, Y = X[perm], Y[perm]
#     Z = np.concatenate((X, Y))
#     KX, KY, KXY, KYX = (kernel_func(X), kernel_func(Y), kernel_func(X,Y),
#                          kernel_func(Y,X))
#     KZ = kernel_func(Z)

#     blockMMD = np.zeros((num_blocks,)) 
#     sigMMD = np.zeros((num_blocks,))
#     for i in range(b):
#         for j in range(i+1, b):
#             idx11, idx12 = num_blocks*i, num_blocks*(i+1)
#             idx21, idx22 = num_blocks*j, num_blocks*(j+1)
#             idx1 = np.arange(idx11, idx12)
#             idx2 = np.arange(idx21, idx22)
#             # update the block-MMD value 
#             blockMMD += KX[idx1, idx2]
#             blockMMD += KY[idx1, idx2]
#             blockMMD -= KXY[idx1, idx2] 
#             blockMMD -= KYX[idx1, idx2]
#             # update the variance estiamte 
#             idx1X = np.random.permutation(2*n)[:num_blocks]
#             idx2X = np.random.permutation(2*n)[:num_blocks]
#             idx1Y = np.random.permutation(2*n)[:num_blocks]
#             idx2Y = np.random.permutation(2*n)[:num_blocks]
#             sigMMD += KZ[idx1X, idx2X]
#             sigMMD += KZ[idx1Y, idx2Y]
#             sigMMD -= KZ[idx1X, idx2Y]
#             sigMMD -= KZ[idx1Y, idx2X]
#     # get the statistic
#     blockMMD /= (b*(b-1))
#     stat = blockMMD.mean()
#     # get the standard deviation 
#     sigMMD /= (b*(b-1))
#     std = (1/sqrt(num_blocks))*sigMMD.std()
#     return stat, std


def BlockMMDSquared(X, Y, kernel_func, b=2, perm=None, biased=True, 
                    return_sig=False):
    # sanity checks 
    n, m = len(X), len(Y) 
    n = min(n, m)
    r = n%b 
    n = int(n-r) # drop the n%b terms 
    X, Y = X[:n], Y[:n] 
    # obtain the gram matrix 
    if perm is None:
        perm = np.arange(n) 
    X, Y = X[perm], Y[perm]
    KX, KY, KXY = kernel_func(X), kernel_func(Y), kernel_func(X, Y)
    KYX = kernel_func(Y, X)
    # compute the block mmd statistic
    num_blocks = n//b 
    blockMMD = np.zeros((num_blocks,)) 
    sigMMD =  np.zeros((num_blocks,))
    tempK = np.zeros((b,b))
    Z = np.concatenate((X, Y))
    for i in range(num_blocks):
        idx0, idx1 = b*i, b*(i+1)
        if biased:
            tempK += (KX[idx0:idx1, idx0:idx1])
            tempK += (KY[idx0:idx1, idx0:idx1])
            tempK -= (KXY[idx0:idx1, idx0:idx1])
            tempK -= (KYX[idx0:idx1, idx0:idx1])
        else:
            Xi = X[idx0:idx1]
            Yi = Y[idx0:idx1]
            blockMMD[i] = TwoSampleMMDSquared(Xi, Yi, kernel_func, unbiased=True) 
            idx1X = np.random.permutation(2*n)[:b]
            idx1Y = np.random.permutation(2*n)[:b]
            sigMMD[i] = TwoSampleMMDSquared(
                X=Z[idx1X], Y=Z[idx1Y], kernel_func=kernel_func, unbiased=True
            )
    if biased:
        stat = (1/num_blocks)*(tempK.mean())
        if return_sig:
            raise Exception('Not computed the std for biased version')
        else:
            return stat
    else:
        stat = blockMMD.mean()
        if return_sig:
            sig = (1/sqrt(num_blocks))*sigMMD.std()
            return stat, sig
        else:
            return stat


###====================Bandwidth selection utils=======================
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
        

###====================Test Calibration methods=======================
def get_bootstrap_threshold(X, Y, kernel_func, statfunc, alpha=0.05,
                            num_perms=500, progress_bar=False,
                            return_stats=False):
    """
        Return the level-alpha rejection threshold for the statistic 
        computed by the function handle stat_func using num_perms 
        permutations. 
    """
    assert len(X.shape)==2
    # concatenate the two samples 
    Z = np.vstack((X,Y))
    # assert len(X)==len(Y)
    n,  n_plus_m = len(X), len(Z)
    # kernel matrix of the concatenated data
    KZ = kernel_func(Z, Z) # 
    
    # original_statistic = statfunc(X, Y, kernel_func)
    perm_statistics = np.zeros((num_perms,))

    range_ = tqdm(range(num_perms)) if progress_bar else range(num_perms)
    for i in range_:
        perm = np.random.permutation(n_plus_m)
        X_, Y_ = Z[perm[:n]], Z[perm[n:]] 
        stat = statfunc(X_, Y_, kernel_func)
        perm_statistics[i] = stat

    # obtain the threshold
    perm_statistics = np.sort(perm_statistics) 

    i_ = int(num_perms*(1-alpha)) 
    threshold = perm_statistics[i_]
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

###====================Misc uitls=======================
def predict_power(PowerDA, alpha=0.05):
    """
    Predict the power of MMD test using the power of cMMD test
    \Phi( z_{\alpha} + \sqrt{2}( \Phi^{-1}(Power) - z_{\alpha}))
    Equation (9) of https://arxiv.org/pdf/2211.14908.pdf
    """
    normal = stats.norm
    z_a = normal.ppf(alpha) 
    sqrt2 = sqrt(2)
    temp = z_a + sqrt2*np.array([
        (normal.ppf(p) - z_a) for p in PowerDA
    ])
    power = normal.cdf(temp) 
    return power

def GaussianVector(mean, cov, n, seed=None):
    # mean = mean.numpy()
    # cov = cov.numpy()
    if seed is not None:
        np.random.seed(seed)
    rv = stats.multivariate_normal(mean, cov) 
    X = rv.rvs(size=n)
    # X = torch.from_numpy(X_).float()
    return X


def DirichletVector(d, n, Alpha=None):
    if Alpha is None:
        Alpha = np.ones((d,))
    X = np.random.dirichlet(alpha=Alpha, size=n) 
    # X = torch.from_numpy(X_).float()
    return X 

# Get the resampled version of empirical variance using a function handle
def get_resampled_std(X, Y, stat_func, kernel_func=None, samples=200):
    if kernel_func is None:
        kernel_func = RBFkernel1

    nx, ny = len(X), len(Y) 
    stat_vals = np.zeros((samples, ))
    for i in range(samples):
        idxX = np.random.randint(0, nx, (nx,))
        idxY = np.random.randint(0, ny, (ny,))
        X_ = X[idxX]
        Y_ = Y[idxY] 
        stat_vals[i] = stat_func(X_, Y_, kernel_func)
    std = stat_vals.std()
    return std 


# Get the resampled version of empirical variance using observation vector
def get_bootstrap_std(obs, num_bootstrap=200):
    vals = np.zeros((num_bootstrap,))
    N = len(obs)
    for i in range(num_bootstrap):
        idx = np.random.choice(a=N, size=(N,))
        # idx = torch.randint(low=0, high=N, size=(N,)) 
        vals[i] = (obs[idx]).mean() 
    return vals.std()
