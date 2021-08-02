from scipy.linalg.lapack import dsysv
from joblib import Parallel, delayed
from datetime import datetime
from proxcd import glasso
#from testimp import glasso
import scipy.linalg as sla
import pandas as pd
import numpy as np


def pglVAR(endog, exog=None, L=1, regconst_max=1., regconst_stps=101,
           regconst_min=0., stp_dec=False,
           gliter=100, gltol=1e-4, n_jobs=12):
    """fit VAR with group-lasso over vars and lags

    Parameters
    ----------
    endog : pandas df
        T x N dimensional array of endogenous variables
    exog : pandas df
        T x M dimensional array of exogenous variables
    L : scalar
        number of lags
    regconst : scalar
        regularizing constant for group lasso

    Returns
    -------
    B : pandas df
        ((N + M) * L) x N coef matrix
    """

    # shape
    T, N = endog.shape
    if exog is not None:
        T, M = exog.shape
    else:
        M = 0
    P = (N + M) * L

    # form lag
    cols = list(endog.columns)
    data_l = []
    for c in cols:
        for l in range(1, L+1):
            df = endog[c].shift(l)
            df.name = "%s_l%d" % (c, l)
            data_l.append(df)
    if exog is not None:
        cols = list(exog.columns)
        for c in cols:
            for l in range(1, L+1):
                df = exog[c].shift(l)
                df.name = "%s_l%d" % (c, l)
                data_l.append(df)
    YX_df = pd.concat(data_l, axis=1).iloc[L:,:]
    endog = endog.iloc[L:,:]

    # form starting values for B
    B = np.zeros(shape=((N + M) * L, N))
    for i, c in enumerate(endog.columns):
        B[:,i] = sla.lstsq(YX_df.values, endog[c].values)[0]
    Bunreg = np.array(B, order="F")

    # form values and vectorized endog
    YX = YX_df.values
    Yc = endog.values

    # form YY, XX, vYvY, XvY
    YXYX = np.array(YX.T.dot(YX), order="F")
    YXYc = np.array(YX.T.dot(Yc), order="F")
    stp_size = np.linalg.eigvalsh(YXYX)[-1] / T * (1 + 10e-6)

    # build regconst path

    # get optimal regconst values given data
    if regconst_max is None:
        regconst_max = np.max(np.abs(YXYc)) / T
    ind = np.logspace(0, 1, regconst_stps)
    ind -= 1
    ind /= ind.max()
    ind *= regconst_max
    ind += regconst_min
    ind = ind[1:]
    if stp_dec:
        regconst_path = np.flip(ind)
    else:
        regconst_path = ind

    # build coefs in parallel
    B_l = []
    B_l.append(Bunreg)
    B_l.extend(Parallel(n_jobs=n_jobs)(
                    delayed(pfit)(regconst, Bunreg, YXYX, YXYc, stp_size,
                                  T, P, N, M, L, gliter, gltol)
                    for regconst in regconst_path))

    var_l = list(YX_df.columns)
    return np.array(B_l), regconst_path, var_l


def pfit(regconst, Bunreg, YXYX, YXYc, stp_size, T, P, N, M, L,
         gliter, gltol):
    """fit glasso for particular fit"""

    # form reg-const array and gradient
    regconst_arr = np.array(np.ones(P) * regconst, order="F")
    grad = np.array((YXYX.dot(Bunreg) - YXYc) / T, order="F")
    # fit glasso
    B = glasso(Bunreg, YXYX / T, grad, stp_size, regconst_arr, N, M, L,
               gliter, gltol)
    B = np.array(B, order="F")

    return B
