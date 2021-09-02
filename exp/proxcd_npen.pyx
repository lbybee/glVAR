from numpy cimport float64_t, ndarray, complex128_t
from numpy import log as nplog
from numpy import square, cov
from numpy import (identity, dot, kron, pi, sum, zeros_like, ones,
                   asfortranarray)
from numpy.linalg import pinv, norm, svd
cimport cython
cimport numpy as cnp
from libc.math cimport sin, cos, acos, exp, sqrt
from cython cimport double

cnp.import_array()

# included in Cython numpy headers
from numpy cimport PyArray_ZEROS
from scipy.linalg.cython_blas cimport sdot, ddot
from scipy.linalg.cython_blas cimport sasum, dasum
from scipy.linalg.cython_blas cimport saxpy, daxpy
from scipy.linalg.cython_blas cimport snrm2, dnrm2
from scipy.linalg.cython_blas cimport scopy, dcopy
from scipy.linalg.cython_blas cimport sscal, dscal
from scipy.linalg.cython_blas cimport srotg, drotg
from scipy.linalg.cython_blas cimport srot, drot
from scipy.linalg.cython_blas cimport sgemv, dgemv
from scipy.linalg.cython_blas cimport sger, dger
from scipy.linalg.cython_blas cimport sgemm, dgemm
from scipy.linalg.cython_blas cimport sgemv, dgemv
from libc.stdio cimport printf


ctypedef float64_t DOUBLE
ctypedef complex128_t dcomplex
cdef int FORTRAN = 1

#cdef extern from "math.h":
#    double log(double x)

cpdef enum BLAS_Order:
    RowMajor  # C contiguous
    ColMajor  # Fortran contiguous


cpdef enum BLAS_Trans:
    NoTrans = 110  # correspond to 'n'
    Trans = 116    # correspond to 't'


cdef void _axpy(int n, double alpha, double *x, int incx,
                double *y, int incy) nogil:
    """y := alph*x + y"""

    daxpy(&n, &alpha, x, &incx, y, &incy)


cdef void _dgemv(char trans, int m, int n, double alpha, double *A,
                 int lda, double *x, int incx, double beta, double *y,
                 int incy) nogil:
    """y := alpha*A*x + beta*y"""

    dgemv(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy)


cdef void _dgemm(char transa, char transb, int m, int n, int k,
                 double alpha, double *A, int lda, double *B,
                 int ldb, double beta, double *C, int ldc) nogil:
    """y := alpha*A*x + beta*y"""

    dgemm(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb,
          &beta, C, &ldc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double glasso_iter(double[::1,:] B, double[::1,:] YXYX,
                        double stp_size, double[::1,:] grad,
                        double[::1] prox, double[::1] regconst,
                        int N, int M, int L, int NL, int NML) nogil:

    cdef:
        int ni = 0
        int nj = 0
        int nq = 0
        int m = 0
        int l = 0
        int lj = 0
        double B_nml
        double gprox
        double score_num = 0
        double score_den = 0

    # iterate over exog vars
    for ni in range(N):

        gprox = 0

        # iterate over lags
        for l in range(L):
            # TODO replace with lapack routine
            for nj in range(N):

                # add the NEGATIVE gradient - (YXYX B - YXYc)
                prox[l*N+nj] = (B[ni*L+l,nj] * stp_size -
                                grad[ni*L+l,nj])

        # generate slice fitted values
        for nj in range(N):
            _dgemv("n", NML, L, -1., &YXYX[0, ni*L], NML,
                   &B[ni*L,nj], 1, 1., &grad[0,nj], 1)

        for l in range(L):
            # TODO replace with lapack routine
            for nj in range(N):

                B_nml = B[ni*L+l,nj]
                B[ni*L+l,nj] = prox[l*N+nj] / stp_size

                # update score
                score_num += (B[ni*L+l,nj] - B_nml) ** 2
                score_den += B_nml ** 2

        # generate slice fitted values
        for nj in range(N):
            _dgemv("n", NML, L, 1., &YXYX[0, ni*L], NML,
                   &B[ni*L,nj], 1, 1., &grad[0,nj], 1)

    # iterate over endog vars
    for m in range(M):

        gprox = 0

        # iterate over factors
        for l in range(L):
            for nj in range(N):

                # add the NEGATIVE gradient - (YXYX B - YXYc)
                prox[l*N+nj] = (B[N*L+m*L+l,nj] * stp_size -
                                grad[N*L+m*L+l,nj])
                gprox += prox[l*N+nj] ** 2

        # generate slice fitted values
        for nj in range(N):
            _dgemv("n", NML, L, -1., &YXYX[0, N*L+m*L], NML,
                   &B[N*L+m*L,nj], 1, 1., &grad[0,nj], 1)

        # build group penalty
        gprox = (1. - (regconst[N+m] / sqrt(gprox)))
        gprox = max(0, gprox)

        for l in range(L):
            for nj in range(N):

                B_nml = B[N*L+m*L+l,nj]
                B[N*L+m*L+l,nj] = prox[l*N+nj] * gprox / stp_size

                # update score
                score_num += (B[N*L+m*L+l,nj] - B_nml) ** 2
                score_den += B_nml ** 2

        # generate slice fitted values
        for nj in range(N):
            _dgemv("n", NML, L, 1., &YXYX[0, N*L+m*L], NML,
                   &B[N*L+m*L,nj], 1, 1., &grad[0,nj], 1)

    return score_num / score_den


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def glasso(double[::1,:] B, double[::1,:] YXYX, double[::1,:] grad,
           double stp_size, double[::1] regconst,
           int N, int M, int L, int gliter, double gltol,
           double tbeta=0.98):
    """Cython glasso implementation

    B : ((N + M) * L) x N

    YXYX : ((N + M) * L) x ((N + M) * L)

    grad : ((N + M) * L) x N

    stp_size : (N + M) * L

    regconst : (N + M)
    """

    cdef:
        int i = 0
        int NL = N * L
        int NML = (N + M) * L

    cdef cnp.npy_intp pshape[2]
    pshape[0] = <cnp.npy_intp> NL

    cdef:
        double [::1] prox = PyArray_ZEROS(1, pshape,
                                          cnp.NPY_DOUBLE,
                                          FORTRAN)

    # iterations
    with nogil:
        for i in range(gliter):

            # update est
            score = glasso_iter(B, YXYX, stp_size, grad, prox,
                                regconst, N, M, L, NL, NML)

            printf("iter: %d score: %f\n", i, score)

#            # backtracking line search update
#            stp_size = stp_size * tbeta

            # return based on score
            if score < gltol:
                break


    if gliter == (i + 1):
        print("Didn't converge")

    return B
