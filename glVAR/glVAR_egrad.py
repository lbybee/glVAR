from sklearn.base import BaseEstimator
from scipy.linalg.lapack import dsysv
from joblib import Parallel, delayed
from statsmodels.tsa.api import VAR
from proxcdVAR_egrad import glassoVAReg
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.linalg as sla
import pandas as pd
import numpy as np



class glVAReg(BaseEstimator):
    """
    The core methods to implement a VAR with a group-lasso penalty

    Parameters
    ----------
    endog : pandas DataFrame or numpy array
        data-frame containing core variables for VAR
        This set of variables will be used as the LHS in the estimation
    exog : pandas DataFrame, numpy array or None
        if provided, this consists of an additional set of variables
        included in the RHS
    lags : scalar
        number of lags for VAR

    Attributes
    ----------
    endog : pandas DataFrame or numpy array
        see Parameters
    exog : pandas DataFrame, numpy array or None
        see Parameters
    YX_df : pandas DataFrame or numpy array
        expanded RHS including lags
    YXYX : numpy array (fortran ordering)
        inner product of RHS variables with themselves
    YXYc : numpy array (fortran order)
        inner product of RHS variables with LHS variables
    stp_size : scalar
        step-size for group lasso iterations
    T, N, M, lags, P : scalar
        Additional dimensionality attributes
    """

    def __init__(self, endog, exog=None, lags=1):

        # we need to standardize the data for selection but maintain
        # nstd for the IRF analaysis
        endog_nstd = endog.copy()
        endog = (endog - endog.mean()) / endog.std()
        if exog is not None:
            exog_nstd = exog.copy()
            exog = (exog - exog.mean()) / exog.std()
        else:
            exog_nstd = None

        # shape
        T, N = endog.shape
        if exog is not None:
            T, M = exog.shape
        else:
            M = 0
        P = (N + M) * lags

        # form lag
        cols = list(endog.columns)
        data_l = []
        for c in cols:
            for l in range(1, lags+1):
                df = endog[c].shift(l)
                df.name = "%s_l%d" % (c, l)
                data_l.append(df)
        if exog is not None:
            cols = list(exog.columns)
            for c in cols:
                for l in range(1, lags+1):
                    df = exog[c].shift(l)
                    df.name = "%s_l%d" % (c, l)
                    data_l.append(df)
        YX_df = pd.concat(data_l, axis=1).iloc[lags:,:]
        endog = endog.iloc[lags:,:]

        # form values and vectorized endog
        YX = YX_df.values
        Yc = endog.values

        # form YY, XX, vYvY, XvY
        YXYX = np.array(YX.T.dot(YX), order="F")
        diag = np.diag(YXYX)
        ndiag = YXYX - diag * np.eye(YXYX.shape[0])
        ndiag = np.sum(np.abs(ndiag), axis=1)
        ind = np.abs(diag) > ndiag
        print(ind, np.sum(ind), np.sum(ind != 1))
        YXYc = np.array(YX.T.dot(Yc), order="F")
        stp_size = np.linalg.eigvalsh(YXYX)[-1] / (T * N * lags) * (1 + 1e-6)

        # store kept values
        self.endog = endog
        self.exog = exog
        self.YX_df = YX_df
        self.endog_nstd = endog_nstd
        self.exog_nstd = exog_nstd
        self.YXYX = YXYX
        self.YXYc = YXYc
        self.stp_size = stp_size
        self.T = T
        self.N = N
        self.M = M
        self.lags = lags
        self.P = P


    def fit(self, regconst, regconst_arr=None, endog_pen=True, **glkwds):
        """fit group lasso VAR for single reg-const

        Parameters
        ----------
        regconst : scalar
            regularizing constant for single group lasso fit
        regconst_array : numpy array or None
            vector of weights for each group
        endog_pen : bool
            whether the LHS variables are penalized
        """

        # pull relevant values
        T = self.T
        N = self.N
        M = self.M
        P = self.P
        lags = self.lags
        YXYX = self.YXYX
        YXYc = self.YXYc
        stp_size = self.stp_size

        # handle endog penalty
        if regconst_arr is None and endog_pen is False:
            regconst_arr = np.array([0] * N + [1] * M)

        # initialize B starting value
        B = np.zeros(shape=((N + M) * lags, N), order="F")

        # fit glasso
        B = _fitgl(regconst, B, YXYX, YXYc, stp_size, T, P, N, M, lags,
                   regconst_arr, **glkwds)

        return B


    def fit_path(self, regconst_arr=None, endog_pen=True,
                 regconst_max=1., regconst_stps=101, regconst_min=0.,
                 stp_dec=False, n_jobs=1, **glkwds):
        """fit group lasso for path of reg-constants

        Parameters
        ----------
        regconst_array : numpy array or None
            vector of weights for each group
        endog_pen : bool
            whether the LHS variables are penalized
        """

        # pull relevant values
        T = self.T
        N = self.N
        M = self.M
        P = self.P
        lags = self.lags
        YXYX = self.YXYX
        YXYc = self.YXYc
        stp_size = self.stp_size

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
        ind *= (regconst_max - regconst_min)
        ind += regconst_min
        ind = ind[1:]
        if stp_dec:
            regconst_path = np.flip(ind)
        else:
            regconst_path = ind

        # initialize B starting value
        B = np.zeros(shape=((N + M) * lags, N), order="F")

        # build coefs in parallel
        B = Parallel(n_jobs=n_jobs)(
                  delayed(_fitgl)(regconst, B, YXYX, YXYc,
                                  stp_size, T, P, N, M, lags,
                                  regconst_arr, **glkwds)
                  for regconst in regconst_path)
        B = np.array(B)
        var_l = list(self.YX_df.columns)

        return B, regconst_path, var_l


    def predict(self):
        """generated predictions based on group-lasso fit"""

        return self


    def fit_IRF(self, B_gl, irf_horizon=12):
        """Given a selection indicator from the group-lasso form IRF"""

        # extract relevant values
        endog = self.endog_nstd
        exog = self.exog_nstd
        YX_df = self.YX_df_nstd
        lags = self.lags

        # constrain to non-zero RHS
        YX_df_nz = YX_df.iloc[:,B_gl.sum(axis=1) != 0]
        cols = [c for c in YX_df_nz.columns]
        nexog = exog[[c for c in exog.columns if ("%s_l1" % c) in cols]]
        fendog = pd.concat([endog, nexog], axis=1).dropna()
        fendog = pd.concat([exog[["71"]],
                            endog[["lsp_ld", "ffr", "lemp", "lgdp"]]],
                            axis=1).dropna()

        # TODO should be able to implement from scratch
        # this introduces overhead but is easier
        mod = VAR(fendog)
        fit = mod.fit(lags)
        irf = fit.irf(irf_horizon)
        irf.plot()
        plt.savefig("test.png")
        irf = irf.irfs

        return irf


    def fit_IRF_CI(self, B_gl, regconst, irf_horizon=12,
                   bsiter=500, n_jobs=1, regconst_arr=None,
                   endog_pen=True, **glkwds):
        """fit bootstrap IRFs to form confidence intervals"""

        # extract relevant values
        endog = self.endog_nstd
        exog = self.exog_nstd
        YX_df = self.YX_df
        lags = self.lags

        # constrain to non-zero RHS
        YX_df_nz = YX_df.iloc[:,B_gl.sum(axis=1) != 0]
        cols = [c for c in YX_df_nz.columns]
        nexog = exog[[c for c in exog.columns if ("%s_l1" % c) in cols]]
        fendog = pd.concat([endog, nexog], axis=1).dropna()

        # fit main active set VAR
        mod = VAR(fendog)
        fit = mod.fit(lags)
        resid = fit.resid
        pred = fendog - resid
        resid = resid[[c for c in resid.columns if c in endog.columns]]
        pred = pred[[c for c in pred.columns if c in endog.columns]]
        rcov = np.cov(resid.T)

        # using bootstrap refit group-lasso procedure
        irf_l = Parallel(n_jobs=n_jobs)(
                    delayed(self.fit_IRF_part)(pred, rcov, endog, exog,
                                               lags, regconst,
                                               regconst_arr, endog_pen,
                                               irf_horizon, **glkwds)
                    for i in range(bsiter))

        return irf_l


    def fit_IRF_part(self, pred, rcov, endog, exog, lags, regconst,
                     regconst_arr, endog_pen, irf_horizon, **glkwds):
        """fit IRF for boostrapped errors"""

        # resample noise
        shp = pred.shape
        noise = np.random.multivariate_normal(mean=np.zeros(shp[1]),
                                              cov=rcov, size=shp[0])
        noise = pd.DataFrame(noise, index=pred.index, columns=pred.columns)
        endog = pred + noise
        endog = endog.dropna()
        exog_f = exog[exog.index.isin(endog.index)]

        # form lags
        endog_nstd = endog.copy()
        endog = (endog - endog.mean()) / endog.std()
        exog_nstd = exog_f.copy()
        exog = (exog_f - exog_f.mean()) / exog_f.std()

        # shape
        T, N = endog.shape
        if exog is not None:
            T, M = exog.shape
        else:
            M = 0
        P = (N + M) * lags

        # form lag
        cols = list(endog.columns)
        data_l = []
        for c in cols:
            for l in range(1, lags+1):
                df = endog[c].shift(l)
                df.name = "%s_l%d" % (c, l)
                data_l.append(df)
        if exog is not None:
            cols = list(exog.columns)
            for c in cols:
                for l in range(1, lags+1):
                    df = exog[c].shift(l)
                    df.name = "%s_l%d" % (c, l)
                    data_l.append(df)
        YX_df = pd.concat(data_l, axis=1).iloc[lags:,:]
        endog = endog.iloc[lags:,:]

        # form values and vectorized endog
        YX = YX_df.values
        Yc = endog.values

        # form YY, XX, vYvY, XvY
        YXYX = np.array(YX.T.dot(YX), order="F")
        YXYc = np.array(YX.T.dot(Yc), order="F")
        stp_size = np.linalg.eigvalsh(YXYX)[-1] / (T * N * lags) * (1 + 1e-6)

        # handle endog penalty
        if regconst_arr is None and endog_pen is False:
            regconst_arr = np.array([0] * N + [1] * M)

        # initialize B starting value
        B = np.zeros(shape=((N + M) * lags, N), order="F")
        for i, c in enumerate(endog.columns):
            coef = sm.OLS(endog[c], YX_df).fit().params.values
            B[:,i] = coef

        # fit glasso
        B_gl = _fitgl(regconst, B, YXYX, YXYc, stp_size, T, P, N, M, lags,
                      regconst_arr, **glkwds)

        # fit active set VAR and form IRFs
        YX_df_nz = YX_df.iloc[:,B_gl.sum(axis=1) != 0]
        cols = [c for c in YX_df_nz.columns]
        nexog = exog[[c for c in exog.columns if ("%s_l1" % c) in cols]]
        fendog = pd.concat([endog, nexog], axis=1).dropna()
        mod = VAR(fendog)
        fit = mod.fit(lags)
        irf = fit.irf(irf_horizon).irfs

        return irf


def _fitirf(rcov, B_as, irf_horizon, lags):
    """helper function to generate IRF"""

    # get dimensions
    Nbig = rcov.shape[0]
    Nbigcomp = B_as.shape[0]

    # reshape B_as to update more easily
    Bcomp = np.eye(Nbigcomp)
    for k in range(Nbig):
        for l in range(lags):
            Bcomp[:Nbig,(Nbig*l)+k] = B_as[(lags*k)+l,:]

    # build shock sizes
    rshock = np.zeros((Nbigcomp, Nbigcomp))
    rshock[:Nbig,:Nbig] = np.linalg.cholesky(rcov) * np.eye(Nbig)

    # form IRF
    Impmat = np.eye(Nbigcomp)
    IR_l = []
    for h in range(irf_horizon):
        IRbig = Impmat.dot(rshock)
        IR_l.append(IRbig[:Nbig,:Nbig])
        Impmat = Impmat.dot(Bcomp)
        print(Impmat, h)
    IR = np.array(IR_l)

    return IR


def _fitas(fendog, YX_df_nz):
    """fit active set VAR given existing glasso coefs"""

    # fit active set VAR
    B_as_l = []
    for c in fendog.columns:
        B_as = sla.lstsq(YX_df_nz, fendog[c])[0]
        B_as_l.append(B_as)
    B_as = np.array(B_as_l).T

    # form fitted values
    pred = YX_df_nz.dot(B_as)
    pred.columns = fendog.columns

    return B_as, pred


def _fitgl(regconst, Binit, YXYX, YXYc, stp_size, T, P, N, M, lags,
           regconst_arr, **glkwds):
    """helper function to fit glasso for given config"""

    # form reg-const array
    if regconst_arr is None:
        regconst_arr = np.array(np.ones(N + M) * regconst, order="F")
    else:
        regconst_arr = np.array(regconst_arr * regconst, order="F")

    # form gradient
    grad = np.array((YXYX.dot(Binit) - YXYc) / T, order="F")

    # fit glasso
    B = glassoVAReg(Binit, Binit, YXYX / T, grad, stp_size, regconst_arr,
                    N, M, lags, **glkwds)
    B = np.array(B, order="F")

    return B
