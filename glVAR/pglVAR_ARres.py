from scipy.linalg.lapack import dsysv
from joblib import Parallel, delayed
from datetime import datetime
from proxcd_ARres import glasso
#from testimp import glasso
import statsmodels.api as sm
import scipy.linalg as sla
import pandas as pd
import numpy as np


def pglVAR_ARres(endog, exog=None, L=1, regconst_max=1., regconst_stps=101,
                 regconst_min=0., regconst_arr=None, stp_dec=False,
                 gliter=100, gltol=1e-4, n_jobs=12, endog_pen=True,
                 ar_order="first"):
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

    # if we calc the AR resids first do that here
    if ar_order == "first":
        for c in endog:
            exogt_l = []
            for l in range(1, L + 1):
                val = endog[c].shift(l)
                val.name = "%s_L%d" % (c, l)
                exogt_l.append(val)
            exogt = pd.concat(exogt_l, axis=1).dropna()
            endogc = endog[c]
            endogc = endogc[endogc.index.isin(exogt.index)]
            exogt = exogt[exogt.index.isin(endogc.index)]
            mod = sm.OLS(endogc, sm.add_constant(exogt))
            fit = mod.fit()
            endog[c] = fit.resid
        endog = endog.dropna()
    elif ar_order != "last":
        raise ValueError("Unknown ar_order: %s" % ar_order)

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

    # if we calc AR resids after forming exog do that here
    if ar_order == "last":
        for c in endog:
            exogt_l = []
            for l in range(1, L + 1):
                val = endog[c].shift(l)
                val.name = "%s_L%d" % (c, l)
                exogt_l.append(val)
            exogt = pd.concat(exogt_l, axis=1).dropna()
            endogc = endog[c]
            endogc = endogc[endogc.index.isin(exogt.index)]
            exogt = exogt[exogt.index.isin(endogc.index)]
            mod = sm.OLS(endogc, sm.add_constant(exogt))
            fit = mod.fit()
            endog[c] = fit.resid
        endog = endog.dropna()
    elif ar_order != "first":
        raise ValueError("Unknown ar_order: %s" % ar_order)
    endog = endog.iloc[L:,:]

    # form starting values for B
    B = np.zeros(shape=((N + M) * L, N))
    for i, c in enumerate(endog.columns):
        B[:,i] = sla.lstsq(YX_df.values, endog[c].values)[0]
        # zero out entries corresponding to pre-fit endog cols
        B[(i*L):((i+1)*L),i] = 0
    Bunreg = np.array(B, order="F")

    # form values and vectorized endog
    YX = YX_df.values
    Yc = endog.values

    # form YY, XX, vYvY, XvY
    YXYX = np.array(YX.T.dot(YX), order="F")
    # zero out rows corresponding to pre-fit endog cols
    YXYc = YX.T.dot(Yc)
    for i, c in enumerate(endog.columns):
        YXYc[(i*L):((i+1)*L),i] = 0
    YXYc = np.array(YXYc, order="F")
    stp_size = np.linalg.eigvalsh(YXYX)[-1] / (T * N * L) * (1 + 1e-6)

    # handle endog penalty
    if regconst_arr is None and endog_pen is False:
        regconst_arr = np.array([0] * N + [1] * M)

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

    B = np.random.normal(size=((N + M) * L, N))

    # build coefs in parallel
    B_l = []
    B_l.append(Bunreg)
    B_l.extend(Parallel(n_jobs=n_jobs)(
                    delayed(pfit)(regconst, Bunreg, YXYX, YXYc, stp_size,
                                  T, P, N, M, L, regconst_arr, gliter, gltol)
                    for regconst in regconst_path))

    var_l = list(YX_df.columns)
    return np.array(B_l), regconst_path, var_l


def pfit(regconst, Bunreg, YXYX, YXYc, stp_size, T, P, N, M, L,
         regconst_arr, gliter, gltol):
    """fit glasso for particular fit"""

    # form reg-const array
    if regconst_arr is None:
        regconst_arr = np.array(np.ones(N + M) * regconst, order="F")
    else:
        regconst_arr = np.array(regconst_arr * regconst, order="F")

    # form gradient
    grad = np.array((YXYX.dot(Bunreg) - YXYc) / T, order="F")

    # fit glasso
    B = glasso(Bunreg, YXYX / T, grad, stp_size, regconst_arr, N, M, L,
               gliter, gltol)
    B = np.array(B, order="F")

    return B