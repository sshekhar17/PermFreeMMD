from math import sqrt 
from functools import partial 
import numpy as np 
import scipy.stats as stats 
from scipy.spatial.distance import cdist, pdist 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
plt.style.use('seaborn-white')

def RBFkernel(x, y=None, bw=1.0, amp=1.0):
    y = x if y is None else y
    dists = cdist(x, y)
    squared_dists = dists * dists 
    k = amp * np.exp( -(1/(2*bw*bw)) * squared_dists ) 
    return k 

def LinearKernel_(x, y=None):
    y = x if y is None else y 
    k = np.einsum('ji, ki -> jk', x, y)
    return k 

def PolynomialKernel_(x, y=None, c=1.0, scale=1.0,  degree=2):
    y = x if y is None else y
    # get the matrix of dot-products 
    D = LinearKernel(x=x, y=y)
    # compute the polynomial kernel 
    assert scale!=0
    k = ((D + c)/scale)**degree 
    return k 



def RBFkernel1(x, y=None, bw=1.0, amp=1.0, pairwise=False):
    """
    Parameters 
        x   : (n, d) torch tensor 
        y   : (m, d) torch tensor
                if y is None, then we set y = x
        bw  : bandwidth  
        amp : amplitude

    Returns
        k   : (n, m) torch tensor         

    Notes
        The kernel implements the function 

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


def LinearKernel(x, y=None):
    y = x if y is None else y 
    k = np.einsum('ji, ki -> jk', x, y)
    return k 

def PolynomialKernel(x, y=None, c=1.0, scale=1.0, 
                        degree=2):
    y = x if y is None else y
    # get the matrix of dot-products 
    D = LinearKernel(x=x, y=y)
    # compute the polynomial kernel 
    assert scale!=0
    k = ((D + c)/scale)**degree 
    return k 


def crossMMD2sampleUnpaired(X, Y, kernel_func):
    """
        Compute the studentized cross-MMD statistic
    """
    n, d = X.shape 
    m, d_ = Y.shape 
    # sanity check 
    assert (d_==d) and (n>=2) and (m>=2) 

    n1, m1 = n//2, m//2 
    n1_, m1_ = n-n1, m-m1

    X1, X2 = X[:n1], X[n1:]
    Y1, Y2 = Y[:m1], Y[m1:]

    Kxx = kernel_func(X1, X2) 
    Kyy = kernel_func(Y1, Y2)

    Kxy = kernel_func(X1, Y2) 
    Kyx = kernel_func(Y1, X2)

    # compute the numerator 
    Ux = Kxx.mean() - Kxy.mean()
    Uy = Kyx.mean() - Kyy.mean()  
    U = Ux - Uy
    # compute the denominator 
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
        # term1 -= torch.trace(Kxx)
        # term2 -= torch.trace(Kyy)
        term1 -= np.trace(Kxx)
        term2 -= np.trace(Kyy)
        MMD_squared = (term1/(n*(n-1)) + term2/(m*(m-1)) - term3)
    else:
        MMD_squared = term1/(n*n) + term2/(m*m) - term3 
    if return_float:
        return MMD_squared
    else:
        return MMD_squared 

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

def get_median_bw_utils(Z=None, X=None, Y=None):
    if Z is None:
        assert (X is not None) and (Y is not None)
        Z = np.concatenate([X, Y], axis=0)
    dists_ = pdist(Z)
    sig = np.median(dists_)
    return sig



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
        bw = get_median_bw_utils(X=X, Y=Y)
    gamma = 1 / (2 * bw**2)
    rbf = lambda A, B: np.exp(-gamma * ((A - B) ** 2).sum(axis=1))
    mmd2_ = (rbf(X[:n:2], X[1:n:2]) + rbf(Y[:n:2], Y[1:n:2])
          - rbf(X[:n:2], Y[1:n:2]) - rbf(X[1:n:2], Y[:n:2]))/2 
    stat = mmd2_.mean()
    if return_scaled:
        sig = mmd2_.std()
        stat = sqrt(n//2)*stat/sig 
    return stat

def BlockMMDSquaredNew(X, Y, kernel_func, b=2, perm=None, biased=False):
    """
        Compute the block mmd squared statistic 
        This is the python translation of the matlab code written by 
        W. Zaremba at this link:
        https://github.com/wojzaremba/btest/blob/master/btest.m
    """
    # sanity checks 
    n, m = len(X), len(Y) 
    if m>n:
        Y=Y[:n] 
        # m=n # we don't use m later in the function
    elif m<n:
        n=m 
        X=X[:n]
    r = n%b 
    if r!=0: # drop the last r terms in X and Y
        n = int(n-r)
        X, Y = X[:n], Y[:n] 
    # now compute the statistic 
    num_blocks = n//b 
    if perm is None:
        perm = np.arange(n) 
    X, Y = X[perm], Y[perm]
    Z = np.concatenate((X, Y))
    KX, KY, KXY, KYX = (kernel_func(X), kernel_func(Y), kernel_func(X,Y),
                         kernel_func(Y,X))
    KZ = kernel_func(Z)

    blockMMD = np.zeros((num_blocks,)) 
    sigMMD = np.zeros((num_blocks,))
    for i in range(b):
        for j in range(i+1, b):
            idx11, idx12 = num_blocks*i, num_blocks*(i+1)
            idx21, idx22 = num_blocks*j, num_blocks*(j+1)
            idx1 = np.arange(idx11, idx12)
            idx2 = np.arange(idx21, idx22)
            # update the block-MMD value 
            blockMMD += KX[idx1, idx2]
            blockMMD += KY[idx1, idx2]
            blockMMD -= KXY[idx1, idx2] 
            blockMMD -= KYX[idx1, idx2]
            # update the variance estiamte 
            idx1X = np.random.permutation(2*n)[:num_blocks]
            idx2X = np.random.permutation(2*n)[:num_blocks]
            idx1Y = np.random.permutation(2*n)[:num_blocks]
            idx2Y = np.random.permutation(2*n)[:num_blocks]
            sigMMD += KZ[idx1X, idx2X]
            sigMMD += KZ[idx1Y, idx2Y]
            sigMMD -= KZ[idx1X, idx2Y]
            sigMMD -= KZ[idx1Y, idx2X]
    # get the statistic
    blockMMD /= (b*(b-1))
    stat = blockMMD.mean()
    # get the standard deviation 
    sigMMD /= (b*(b-1))
    std = (1/sqrt(num_blocks))*sigMMD.std()
    return stat, std


def BlockMMDSquared(X, Y, kernel_func, b=2, perm=None, biased=True, 
                    return_sig=False):
    # sanity checks 
    n, m = len(X), len(Y) 
    if m>n:
        Y=Y[:n] 
    elif m<n:
        n=m 
        X=X[:n]
    r = n%b 
    if r!=0: # drop the last r terms in X and Y
        n = int(n-r)
        X, Y = X[:n], Y[:n] 
    # now compute the statistic 
    num_blocks = n//b 
    if perm is None:
        perm = np.arange(n) 
    X, Y = X[perm], Y[perm]
    KX, KY, KXY = kernel_func(X), kernel_func(Y), kernel_func(X, Y)
    KYX = kernel_func(Y, X)

    # blockMMD = 0 
    blockMMD = np.zeros((num_blocks,)) 
    sigMMD =  np.zeros((num_blocks,))
    tempK = np.zeros((b,b))
    Z = np.concatenate((X, Y))
    for i in range(num_blocks):
        idx0, idx1 = b*i, b*(i+1)
#        idx = perm[idx0:idx1]
#        Xi, Yi = X[idx], Y[idx] 
#        KX, KY, KXY = kernel_func(Xi), kernel_func(Yi), kernel_func(Xi, Yi)
        if biased:
            tempK += (KX[idx0:idx1, idx0:idx1])
            tempK += (KY[idx0:idx1, idx0:idx1])
            tempK -= (KXY[idx0:idx1, idx0:idx1])
            tempK -= (KYX[idx0:idx1, idx0:idx1])
        else:
            Xi = X[idx0:idx1]
            Yi = Y[idx0:idx1]
            # blockMMD += TwoSampleMMDSquared(Xi, Yi, kernel_func, unbiased=True) 
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
        # stat = (1/num_blocks)*blockMMD
        stat = blockMMD.mean()
        if return_sig:
            sig = (1/sqrt(num_blocks))*sigMMD.std()
            return stat, sig
        else:
            return stat


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


def predict_power(PowerDA, alpha=0.05):
    """
    Predict the power of MMD test using the power of cMMD test
    \Phi( z_{\alpha} + \sqrt{2}( \Phi^{-1}(Power) - z_{\alpha}))
    """
    normal = stats.norm
    z_a = normal.ppf(alpha) 
    sqrt2 = sqrt(2)
    temp = z_a + sqrt2*np.array([
        (normal.ppf(p) - z_a) for p in PowerDA
    ])
    power = normal.cdf(temp) 
    return power

# def CreateGaussianSource(mean1, cov1, mean2, cov2):

#     def Source(n, m=None):
#         m = n if m is None else m 
#         rvX = stats.multivariate_normal(mean=mean1, cov=cov1) 
#         rvY = stats.multivariate_normal(mean=mean2, cov=cov2) 
#         X = rvX.rvs(size=n)
#         Y = rvY.rvs(size=m)
#         return X, Y 
    
#     return Source 


if __name__=='__main__':
    d = 10 
    epsilon = 0.0
    num_trials=500
    n=500

    meanX = np.zeros((d,))
    meanY = np.zeros((d,))
    meanY[:5] = epsilon

    def SourceX(n):
        return GaussianVector(mean=meanX, cov=np.eye(d), n=n)

    def SourceY(n):
        return GaussianVector(mean=meanY, cov=np.eye(d), n=n)

    linmmdstat= np.zeros((num_trials,))
    blockmmd = np.zeros((num_trials))




    for i in tqdm(range(500)):
        X, Y = SourceX(n), SourceY(n)

        # get the block-mmd statistic
        bw = get_median_bw_utils(X=X, Y=Y)
        kernel_func = partial(RBFkernel, bw=bw)
        stat, sig = BlockMMDSquared(X=X, Y=Y, kernel_func=kernel_func, b=10,
                            biased=False, return_sig=True)
        blockmmd[i] = stat/sig
    
        # get the linear mmd statistic
        linmmdstat[i] = LinearMMDrbf(X=X, Y=Y, return_scaled=True)
    plt.hist(linmmdstat, density=True, label='linear', alpha=0.5)
    plt.hist(blockmmd, density=True, label='block', alpha=0.5)
    plt.legend()