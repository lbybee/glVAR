from sklearn.base import BaseEstimator
from scipy.linalg.lapack import dsysv
from joblib import Parallel, delayed
from statsmodels.tsa.vector_ar.util import varsim
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from proxcdVAR import glassoVAR
from datetime import datetime
import scipy.linalg as sla
import pandas as pd
import numpy as np
import copy
import os



class glVAR(BaseEstimator):
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

    def __init__(self, endog, exog=None, lags=1, std=True):

        # group lasso fits need to be std, provide this option
        if std:
            endog = (endog - endog.mean()) / endog.std()
        if exog is not None and std:
            exog = (exog - exog.mean()) / exog.std()

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
        stp_size = np.linalg.eigvalsh(YXYX)[-1] / T * (1 + 1e-6)

        # store kept values
        self.endog = endog
        self.exog = exog
        self.YX_df = YX_df
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
        endog = self.endog
        YX_df = self.YX_df

        # handle endog penalty
        if regconst_arr is None and endog_pen is False:
            regconst_arr = np.array([0] * N + [1] * M)

        # initialize B starting value
        B = np.zeros(shape=((N + M) * lags, N), order="F")
        for i, c in enumerate(endog.columns):
            coef = sla.lstsq(YX_df.values, endog[c].values)[0]
            B[:,i] = coef

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
        endog = self.endog
        YX_df = self.YX_df

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
        for i, c in enumerate(endog.columns):
            coef = sla.lstsq(YX_df.values, endog[c].values)[0]
            B[:,i] = coef

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


    def fit_IRF(self, B_gl, col_order=None, irf_horizon=12):
        """Given a selection indicator from the group-lasso form IRF"""

        # extract relevant values
        endog = self.endog
        exog = self.exog
        YX_df = self.YX_df
        lags = self.lags

        # constrain to non-zero RHS
        YX_df_nz = YX_df.iloc[:,B_gl.sum(axis=1) != 0]
        cols = [c for c in YX_df_nz.columns]
        nexog = exog[[c for c in exog.columns if ("%s_l1" % c) in cols]]
        fendog = pd.concat([endog, nexog], axis=1).dropna()
        if col_order is not None:
            fendog = fendog[col_order]

        # TODO should be able to implement from scratch
        # this introduces overhead but is easier
        mod = VAR(fendog)
        fit = mod.fit(lags)
#        irf = fit.orth_ma_rep(maxn=irf_horizon)
        irf = fit.irf(irf_horizon).irfs

        return irf


    def fit_IRF_CI(self, B_gl, regconst, irf_horizon=12,
                   bsiter=500, n_jobs=1, regconst_arr=None,
                   endog_pen=True, col_order=None,
                   burn=100, **glkwds):
        """fit bootstrap IRFs to form confidence intervals"""

        # extract relevant values
        endog = self.endog
        exog = self.exog
        YX_df = self.YX_df
        lags = self.lags

        # constrain to non-zero RHS
        YX_df_nz = YX_df.iloc[:,B_gl.sum(axis=1) != 0]
        cols = [c for c in YX_df_nz.columns]
        nexog = exog[[c for c in exog.columns if ("%s_l1" % c) in cols]]
        fendog = pd.concat([endog, nexog], axis=1).dropna()
        if col_order is not None:
            fendog = fendog[col_order]
        mod = VAR(fendog)
        fit = mod.fit(lags)
        irf = fit.irf(irf_horizon).orth_irfs

        # using bootstrap refit group-lasso procedure
        res_l = Parallel(n_jobs=n_jobs)(
                    delayed(self.fit_IRF_part)(fit, endog, exog,
                                               lags, regconst,
                                               regconst_arr, endog_pen,
                                               irf_horizon, col_order,
                                               burn, B_gl, **glkwds)
                    for i in range(bsiter))
        irf_l, coef_l, int_l = zip(*res_l)

        return (np.array(irf_l), np.array(coef_l), np.array(int_l),
                irf, fit.coefs, fit.intercept)


    def fit_IRF_part(self, fit, endog, exog, lags, regconst,
                     regconst_arr, endog_pen, irf_horizon, col_order,
                     burn, B_gl, **glkwds):
        """fit IRF for boostrapped errors"""

        # resample noise
#        sim = varsim(coefs=fit.coefs,
#                     intercept=fit.intercept,
#                     sig_u=fit.sigma_u,
#                     steps=fit.nobs + fit.k_ar + burn)
#        sim = pd.DataFrame(sim[burn:],
#                           columns=fit.fittedvalues.columns,
#                           index=endog.index)
        pred = fit.fittedvalues
        resid = fit.resid
        chol = np.linalg.cholesky(resid.cov())
        shp = pred.shape
#        noise = resid.copy()
#        for c in noise.columns:
#            noise[c] = noise[c].sample(frac=1)
        noise = np.random.multivariate_normal(mean=np.zeros(shp[1]),
                                              cov=resid.cov(), size=shp[0])
        noise = pd.DataFrame(noise, index=pred.index, columns=pred.columns)
#        sim = pred + noise
        sim = pred + resid
        asetvars = list(sim.columns)
        end_cols = [c for c in sim.columns if c in endog.columns]
        ex_cols = [c for c in sim.columns if c in exog.columns]

        # build new endog/exog
        endog_f = endog.copy()
        endog_f[end_cols] = sim[end_cols]
        endog_f = endog_f.dropna()
        exog_f = exog.copy()
        exog_f[ex_cols] = sim[ex_cols]
        exog_f = exog_f.dropna()

        # form lags
        endog_std = (endog_f - endog_f.mean()) / endog_f.std()
        exog_std = (exog_f - exog_f.mean()) / exog_f.std()

        # shape
        T, N = endog_std.shape
        if exog_std is not None:
            T, M = exog_std.shape
        else:
            M = 0
        P = (N + M) * lags

        # form lag
        cols = list(endog_std.columns)
        data_l = []
        for c in cols:
            for l in range(1, lags+1):
                df = endog_std[c].shift(l)
                df.name = "%s_l%d" % (c, l)
                data_l.append(df)
        if exog_std is not None:
            cols = list(exog_std.columns)
            for c in cols:
                for l in range(1, lags+1):
                    df = exog_std[c].shift(l)
                    df.name = "%s_l%d" % (c, l)
                    data_l.append(df)
        YX_df = pd.concat(data_l, axis=1).iloc[lags:,:]
        endog_std = endog_std.iloc[lags:,:]

        # form values and vectorized endog
        YX = YX_df.values
        Yc = endog_std.values

        # form YY, XX, vYvY, XvY
        YXYX = np.array(YX.T.dot(YX), order="F")
        YXYc = np.array(YX.T.dot(Yc), order="F")
        stp_size = np.linalg.eigvalsh(YXYX)[-1] / T * (1 + 1e-6)

        # handle endog penalty
        if regconst_arr is None and endog_pen is False:
            regconst_arr = np.array([0] * N + [1] * M)

        # initialize B starting value
        B = np.zeros(shape=((N + M) * lags, N), order="F")
        for i, c in enumerate(endog_std.columns):
            coef = sla.lstsq(YX_df.values, endog_std[c].values)[0]
            B[:,i] = coef

        # fit glasso
        B_gl = _fitgl(regconst, B, YXYX, YXYc, stp_size, T, P, N, M, lags,
                      regconst_arr, **glkwds)

        # prep active set data
        YX_df_nz = YX_df.iloc[:,B_gl.sum(axis=1) != 0]
        cols = [c for c in YX_df_nz.columns]
        nexog = exog_f[[c for c in exog_f.columns if ("%s_l1" % c) in cols]]
        fendog = pd.concat([endog_f, nexog], axis=1).dropna()
        if col_order is not None:
            ncols = [c for c in fendog.columns if c not in col_order]
            col_ordern = [c for c in col_order if c in fendog.columns]
            fendog = fendog[col_ordern + ncols]

        # get map from new endog labels to old
        find = pd.Series(fendog.columns, name="label")
        find.index.name = "ind"
        find = find.reset_index().set_index("label")["ind"]
        oind = pd.Series(asetvars, name="label")
        oind.index.name = "oind"
        oind = oind.reset_index()
        oind["nind"] = oind["label"].map(find)
        oind = oind[["oind", "nind"]].dropna()
        nind = oind["nind"].astype(int).values
        oind = oind["oind"].astype(int).values

        # fit active set VAR and form IRFs
        print(fendog.columns)
        mod = VAR(fendog)
        fit = mod.fit(lags)
        irf = fit.irf(irf_horizon).orth_irfs
#        irf = fit.irf(irf_horizon, var_decomp=chol).orth_irfs
        irft = np.zeros((irf.shape[0], len(asetvars), len(asetvars)))
        for oi, ni in zip(oind, nind):
            irft[:,oi,oind] = irf[:,ni,nind]

        return irft, fit.coefs, fit.intercept


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
    B = glassoVAR(Binit, YXYX / T, grad, stp_size, regconst_arr,
                  N, M, lags, **glkwds)
    B = np.array(B, order="F")

    return B
