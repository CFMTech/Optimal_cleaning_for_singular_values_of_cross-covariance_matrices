"""
Created on Thu Aug 16 17:32:18 2018

@author: florent
"""
import os
import sys
import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sklearn.isotonic import IsotonicRegression

plt.rcParams['text.usetex'] = True
plt.rcParams["legend.fontsize"] = 15
plt.rcParams['axes.labelsize'] = 15


def f0(x):
    return (
        x
        * (3 - x) ** 2
        * ((x - 1.7) ** 2 + 0.1)
        * (((x - 2.6) ** 2 + 1) ** (-1) + 1)
        * (((x - 0.9) ** 2 + 1) ** (-1) + 1)
        * (3.1 - x) ** (-2)
    )


CONSTANT_Z = quad(f0, 0, 3)[0] ** (-1)


def the_fancy_density(x):
    return CONSTANT_Z * f0(x)


def the_final_fancy_density(y):
    min_s, max_s = 0.2, 0.8
    return (3 / (max_s - min_s)) * the_fancy_density(
        -3 * (y - min_s) / (max_s - min_s) + 3
    )


def the_fancy_ran_var():
    min_s, max_s = 0.2, 0.8
    a, b = 0., 3.
    x = np.random.uniform(low=a, high=b)
    y = np.random.uniform()
    while y > the_fancy_density(x):
        x = np.random.uniform(low=a, high=b)
        y = np.random.uniform()
    return min_s + (max_s - min_s) * (3 - x) / 3


def my_int(x):
    return int(round(x))


def Haar_Orthogonal(n=100):
    """Haar distributed orthogonal matrix sampled according to Mezzadri's arXiv:math-ph/0609050 algo"""
    M = np.random.randn(n, n)
    Q, R = np.linalg.qr(M)
    D = np.diag(R)
    D = D / np.abs(D)
    return np.matrix(Q) * np.diag(D)


def Haar_Grassman_Real(n=100, r=100):
    """r first columns of a Haar distributed n by n Haar distributed orthogonal matrix sampled according to Mezzadri's arXiv:math-ph/0609050 algo"""
    return Haar_Orthogonal(n)[:, :r]


def Rectangular_Real_SRT(n=5, p=8, s=np.linspace(0.2, 0.8, 3)):
    r = len(s)
    return Haar_Grassman_Real(n, r) * np.diag(s) * (Haar_Grassman_Real(p, r).T)


def Rectangular_Real_SRT_Reg(n=5, p=8, mins=0.2, Maxs=0.8):
    return Rectangular_Real_SRT(n=n, p=p, s=np.linspace(mins, Maxs, min(n, p)))


def Wishart(n=100, p=100, real=True):
    """Real or complex n by n Wishart matrix normalized in such a way that
    its spectrum converges to [a,b] with density np.sqrt((b-x)*(x-a))/(2*np.pi*c*x)
    with a,b=(1-np.sqrt(c))**2,(1+np.sqrt(c))**2 for c=n/p"""
    if real:
        Y = np.matrix(np.random.randn(n, p))
        return Y * (Y.T) / p
    else:
        Y = 2 ** (-0.5) * np.matrix(np.random.randn(n, p) + 1j * np.random.randn(n, p))
        return Y * (Y.H) / p


def Empirical_Covariance(T=100, C=np.eye(100)):
    """Empirical n by n covariance matrix of a sample of T copies of a centered gaussian vector with covariance C"""
    n = np.shape(C)[0]
    Y = np.matrix(np.random.multivariate_normal(mean=[0] * n, cov=C, size=T))
    return Y.T * Y / T


def Heavy_Tailed_Empirical_Covariance_Matrix(n=100, p=100, alpha=2.5):
    """Empirical covariance matrix of p vectors with size n whose entries
    are iid distributed as \pm U^{-1/alpha}, with U uniform on [0,1]
    and \pm a random sign"""
    X = np.matrix(
        ((np.random.rand(n, p)) ** (-1 / float(alpha)))
        * (2 * np.random.binomial(1, 0.5, size=(n, p)) - 1)
    )
    return X * (X.T) / p


def my_model(T, model=1):
    alpha, beta = .4, .7
    proportion_info = .8
    n, p = my_int(alpha * T), my_int(beta * T)
    m = n + p
    pp = my_int(proportion_info * n)
    if model == 1:
        return (n, p, np.eye(m) + Wishart(m, pp) + np.ones((m, m)) / float(m))
    elif model == 2:
        a, b = .2, .8
        C = Rectangular_Real_SRT_Reg(n, p, a, b)
        return (n, p, np.matrix(np.bmat([[np.eye(n), C], [C.T, np.eye(p)]])))
    elif model == 3:
        return (n, p, np.eye(m))
    elif model == 4:
        al, bet = .1, 1.
        alpha_tail = 15
        return (n, p,
                np.eye(m) + Heavy_Tailed_Empirical_Covariance_Matrix(m, pp, alpha_tail) / float(pp ** al) + np.ones(
                    (m, m)) / float(m ** bet))
    elif model == 5:
        c = 2
        a, b = .2, .8
        C = Rectangular_Real_SRT(n, p, np.linspace(a, b, n) ** c)
        return (n, p, np.matrix(np.bmat([[np.eye(n), C], [C.T, np.eye(p)]])))
    elif model == 6:
        center, widtha, widthb, c, epsi = .5, .1, .3, .8, .01
        s = [min(1 - epsi, max(epsi, center + np.sign(x) * np.abs(x) ** c)) for x in np.linspace(-widtha, widthb, n)]
        C = Rectangular_Real_SRT(n, p, s)
        return (n, p, np.matrix(np.bmat([[np.eye(n), C], [C.T, np.eye(p)]])))
    elif model == 7:
        al, bet = .1, 1.
        alpha_tail = 15
        center, widtha, widthb, c, epsi = .5, .1, .3, .8, .01
        s = [min(1 - epsi, max(epsi, center + np.sign(x) * np.abs(x) ** c)) for x in np.linspace(-widtha, widthb, n)]
        C = Rectangular_Real_SRT(n, p, s)
        TrueTotalCovariance = np.matrix(np.bmat([[np.eye(n), C], [C.T, np.eye(p)]]))
        return (n, p, TrueTotalCovariance + Heavy_Tailed_Empirical_Covariance_Matrix(m, pp, alpha_tail) / float(
            pp ** al) + np.ones((m, m)) / float(m ** bet))
    elif model == 8:
        a, b = .2, .8
        C = Rectangular_Real_SRT(n=n, p=p, s=list(np.linspace(a, b, pp)) + [0] * (n - pp))
        return (n, p, np.matrix(np.bmat([[np.eye(n), C], [C.T, np.eye(p)]])))
    elif model == 9:
        al_Beta_law, beta_Beta_law = 2, 3
        a, b = .2, .8
        C = Rectangular_Real_SRT(n, p,
                                 s=a + (b - a) * np.random.beta(al_Beta_law,
                                                                beta_Beta_law,
                                                                size=n))
        return (n, p, np.matrix(np.bmat([[np.eye(n), C], [C.T, np.eye(p)]])))
    elif model == 10:
        Ech = []
        for i in range(n):
            Ech.append(the_fancy_ran_var())
        C = Rectangular_Real_SRT(n, p, s=np.array(Ech))
        return (n, p, np.matrix(np.bmat([[np.eye(n), C], [C.T, np.eye(p)]])))




def check_matrix(M, n, p):
    assert M is not None
    assert isinstance(n, int)
    assert isinstance(p, int)
    assert isinstance(M, np.matrix), str(type(M))
    assert M.shape == (n, p), f"{M.shape}"


def get_submatrices_of_full_cov_mat(n, p, CZZ):
    check_matrix(CZZ, n + p, n + p)
    CXX = CZZ[np.ix_(range(n), range(n))]
    CYY = CZZ[np.ix_(range(n, n + p), range(n, n + p))]
    CXY = CZZ[np.ix_(range(n), range(n, n + p))]
    return CXX, CYY, CXY


def Coeffs(n, p, U, V, CXXemp, CYYemp):
    for M in [U, CXXemp]:
        check_matrix(M, n, n)
    for M in [V, CYYemp]:
        check_matrix(M, p, p)
    Coeff_A, Coeff_B, Coeff_B_n_to_p = [], [], 0.
    for k in range(n):
        Coeff_A.append((U[:, k].T * CXXemp * U[:, k])[0, 0])
        Coeff_B.append((V[k, :] * CYYemp * (V[k, :].T))[0, 0])
    for k in range(n, p):
        Coeff_B_n_to_p += (V[k, :] * CYYemp * (V[k, :].T))[0, 0]
    return Coeff_A, Coeff_B, Coeff_B_n_to_p


def approx_L_or_imLoimH(z,
                        n,
                        p,
                        T,
                        Coeff_A=None,
                        Coeff_B=None,
                        Coeff_B_n_to_p=None,
                        CXXemp=None,
                        CYYemp=None,
                        CXYemp=None,
                        U=None,
                        s=None,
                        stwo=None,
                        V=None,
                        algo_used=1,
                        return_L=False):
    assert algo_used in (1, 2), f"{algo_used}"
    assert isinstance(n, int) and isinstance(p, int) and n <= p
    if algo_used == 1:
        if any([x is None for x in [Coeff_A, Coeff_B, Coeff_B_n_to_p]]):
            if U is None or V is None:
                check_matrix(CXYemp, n, p)
                U, s, V = np.linalg.svd(CXYemp, full_matrices=True)
            check_matrix(CXXemp, n, n)
            check_matrix(CYYemp, p, p)
            Coeff_A, Coeff_B, Coeff_B_n_to_p = Coeffs(n=n,
                                                      p=p,
                                                      U=U,
                                                      V=V,
                                                      CXXemp=CXXemp,
                                                      CYYemp=CYYemp)
        for c in (Coeff_A, Coeff_B):
            assert isinstance(c, list) and len(c) == n
        assert isinstance(Coeff_B_n_to_p, float), f"{Coeff_B_n_to_p}"
    ztwo = z ** 2
    if stwo is None:
        if s is None:
            check_matrix(CXYemp, n, p)
            s = np.linalg.svd(CXYemp, compute_uv=False)
        stwo = s ** 2
    assert isinstance(stwo, np.ndarray) and len(stwo) == n
    one_over_ztwo_minus_stwo = 1 / (ztwo - stwo)
    TH = np.dot(stwo, one_over_ztwo_minus_stwo)
    fT = float(T)
    H = TH / fT
    if algo_used == 1:
        L = 1 - T / (T + TH - (ztwo * np.dot(Coeff_A, one_over_ztwo_minus_stwo) * (
                np.dot(Coeff_B, one_over_ztwo_minus_stwo) + Coeff_B_n_to_p / ztwo) / (T + TH)))
    else:
        alpha, beta = n / fT, p / fT
        H = TH / fT
        G = (H + alpha) / ztwo
        L = (1 + 2 * H - np.sqrt(1 + 4 * ztwo * G * (G + (beta - alpha) / ztwo) * (1 + H) ** 2)) / (2 + 2 * H)
    return L if return_L else np.imag(L) / np.imag(H)


def RIE_Cross_Covariance(
        CZZemp,
        T,
        n,
        p,
        Return_Sing_Values_only=False,
        Return_Ancient_SV=False,
        Return_New_SV=False,
        Return_Sing_Vectors=False,
        adjust=False,
        return_all=False,
        isotonic=False,
        exponent_eta=0.5,
        c_eta=1,
        algo_used=1
):
    """Flo's algo. We need n\leq p, Etotale needs to be of the type np.matrix"""
    assert algo_used in (1, 2), f"{algo_used}"
    assert isinstance(n, int) and isinstance(p, int) and n <= p
    CXXemp, CYYemp, CXYemp = get_submatrices_of_full_cov_mat(n=n, p=p, CZZ=CZZemp)
    U, s, V = np.linalg.svd(CXYemp, full_matrices=True)
    if algo_used == 1:
        Coeff_A, Coeff_B, Coeff_B_n_to_p = Coeffs(n=n,
                                                  p=p,
                                                  U=U,
                                                  V=V,
                                                  CXXemp=CXXemp,
                                                  CYYemp=CYYemp)
    else:
        Coeff_A, Coeff_B, Coeff_B_n_to_p = None, None, None
    stwo = s ** 2
    eta = c_eta * (n * p * T) ** (-exponent_eta / 3.0)
    new_s = [max(0, approx_L_or_imLoimH(z=s[k] + 1j * eta,
                                        n=n,
                                        p=p,
                                        T=T,
                                        Coeff_A=Coeff_A,
                                        Coeff_B=Coeff_B,
                                        Coeff_B_n_to_p=Coeff_B_n_to_p,
                                        CXXemp=CXXemp,
                                        CYYemp=CYYemp,
                                        CXYemp=CXYemp,
                                        U=U,
                                        V=V,
                                        stwo=stwo,
                                        algo_used=algo_used,
                                        return_L=False)) * s[k] for k in range(n)]
    if adjust:
        new_s = np.array(new_s)
        new_s *= np.sqrt(
            max(0, (T * sum(stwo) - np.trace(CXXemp) * np.trace(CYYemp)) / (T + 1 - 2 * T ** (-1)))) / np.linalg.norm(
            new_s)
    if Return_Sing_Values_only:
        ir = IsotonicRegression()
        new_s_isotonic = ir.fit_transform(np.arange(n)[::-1], np.array(new_s))
        return (s, new_s, new_s_isotonic)
    if return_all:
        ir = IsotonicRegression()
        new_s_isotonic = ir.fit_transform(np.arange(n)[::-1], np.array(new_s))
        return s, U, V, new_s, new_s_isotonic
    if isotonic:
        ir = IsotonicRegression()
        new_s = ir.fit_transform(np.arange(n)[::-1], np.array(new_s))
    U, V = np.matrix(U), np.matrix(V[range(n), :]).T
    res = [U * np.diag(new_s) * (V.T)]
    if Return_Ancient_SV:
        res.append(s)
    if Return_New_SV:
        res.append(new_s)
    if Return_Sing_Vectors:
        res.append(U)
        res.append(V)
    if Return_Ancient_SV or Return_New_SV or Return_Sing_Vectors:
        return tuple(res)
    else:
        return res[0]


def RIE_Covariance(
    Cemp, T, Return_Ancient_Spectrum=False, Return_New_Spectrum=False, adjust_trace=True
):
    """Almost Bouchaud's RIE for the true covariance matrix out of the empirical covariance matrix Cemp made out of a sample of T copies of the signal"""
    lam, U = np.linalg.eigh(Cemp)
    U = np.matrix(U)
    n = len(lam)
    q = n / float(T)
    eta = n ** (-0.5)
    new_lam = []
    for k in range(n):
        oldla = lam[k]
        new_lam.append(
            oldla
            * (
                np.abs(
                    1
                    - q
                    + q * oldla * np.mean(1 / (oldla + eta * 1j - np.delete(lam, k)))
                )
                ** (-2)
            )
        )
    if adjust_trace:
        new_lam = np.array(new_lam) * (sum(lam) / sum(new_lam))
    if Return_Ancient_Spectrum and Return_New_Spectrum:
        return (U * np.matrix(np.diag(new_lam)) * (U.T), lam, new_lam)
    elif Return_Ancient_Spectrum:
        return (U * np.matrix(np.diag(new_lam)) * (U.T), lam)
    elif Return_New_Spectrum:
        return (U * np.matrix(np.diag(new_lam)) * (U.T), new_lam)
    else:
        return U * np.matrix(np.diag(new_lam)) * (U.T)