from scipy.linalg.lapack import dsysv
from datetime import datetime
from proxcd import glasso
#from testimp import glasso
import scipy.linalg as sla
import pandas as pd
import numpy as np


def glVAR(endog, exog=None, L=1, regconst_max=1., regconst_stps=101,
          regconst_min=0., drop_unreg=False, stp_dec=False,
          cstart=False, gliter=100, gltol=1e-4):
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

    # build regconst path
#    # get optimal regconst values given data
#    if regconst_max is None:
#        XXyyt = np.zeros(self.metad["L"])
#        for t in range(self.metad["T"]):
#            ix_mn = self.tind[t]
#            ix_mx = self.tind[t+1]
#            XXyyt += np.square(np.dot(self.X[ix_mn:ix_mx,:].T,
#                                      self.y[ix_mn:ix_mx]))
#        XXyyt = np.sqrt(XXyyt) / (self.X.shape[0])
#        regconst_max = np.max(XXyyt)
    ind = np.logspace(0, 1, regconst_stps)
    ind -= 1
    ind /= ind.max()
    ind *= regconst_max
    ind += regconst_min
    if drop_unreg:
        ind = ind[1:]
    if stp_dec:
        regconst_path = np.flip(ind)
    else:
        regconst_path = ind

    # form lags
    endog_l = []
    for l in range(1, L+1):
        endogl = endog.shift(l)
        endogl.columns = [c + "_l%d" % l for c in endogl.columns]
        endog_l.append(endogl)
    endogl = pd.concat(endog_l, axis=1).iloc[L:,:]
    if exog is not None:
        exog_l = []
        for l in range(1, L+1):
            exogl = exog.shift(l)
            exogl.columns = [c + "_l%d" % l for c in exogl.columns]
            exog_l.append(exogl)
        exogl = pd.concat(exog_l, axis=1).iloc[L:,:]
        YX_df = pd.concat([endogl, exogl], axis=1)
    else:
        YX_df = endogl
    endog = endog.iloc[L:,:]

    # form starting values for B
    B = np.zeros(shape=((N + M) * L, N))
    for i, c in enumerate(endog.columns):
        B[:,i] = sla.lstsq(YX_df.values, endog[c].values)[0]
    B = np.array(B, order="F")

    # form values and vectorized endog
    YX = YX_df.values
    Yc = endog.values

    # form YY, XX, vYvY, XvY
    YXYX = np.array(YX.T.dot(YX), order="F")
    YXYc = np.array(YX.T.dot(Yc), order="F")
    stp_size = np.linalg.eigvalsh(YXYX)[-1] / T * (1 + 10e-6)

    B_l = []
    B_l.append(B)
    var_l = list(YX_df.columns)
    for regconst in regconst_path[1:]:

        # form reg-const array and gradient
        regconst_arr = np.array(np.ones(P) * regconst, order="F")
        grad = np.array((YXYX.dot(B) - YXYc) / T, order="F")
        # fit glasso
        B = glasso(B, YXYX / T, grad, stp_size, regconst_arr, N, M, L,
                   gliter, gltol)
        B = np.array(B, order="F")
        B_l.append(B)

        # form starting values for B
        if cstart:
            B = np.zeros(shape=((N + M) * L, N))
            for i, c in enumerate(endog.columns):
                B[:,i] = sla.lstsq(YX_df.values, endog[c].values)[0]
            B = np.array(B, order="F")

    return np.array(B_l), regconst_path, var_l
