"""
Created on Thu Aug 16 22:42:48 2018

@author: florent
"""
from base_functions import *

N_LGN = 50
T = 500
print(f"T={T} and sample size={N_LGN}: for larger matrices and samples, parallelized jobs are more adapted")
proportion_infos = [0., .1, .2, .3, .4]
alphas_ht = ['5', '2.5', '1.5', '0.5']

tested_cases = (
        [f'lin_model__{prop}' for prop in proportion_infos] +
        ['wishart__2'] +
        [f'heavy_tailed__{al}' for al in alphas_ht] +
        ['mixed_wishart__2'] +
        [f'mixed_heavy_tailed__{al}' for al in alphas_ht]
)


def error_norm(x, x_0, norm='frobenius'):
    assert norm in ('frobenius', 'operator_norm'), norm
    return np.linalg.norm(x - x_0, ord=None if norm == 'frobenius' else 2)


def estimator_comparison(T=500, model_param='lin_model__0.1', alpha=.4, beta=.7, sigmatwo=.5, a=.2, b=.5):
    n, p = my_int(alpha * T), my_int(beta * T)
    model, param = tuple(model_param.split('__'))
    assert model in ('lin_model', 'wishart', 'heavy_tailed', 'mixed_wishart', 'mixed_heavy_tailed'), model
    if model == 'lin_model':
        pp = my_int(float(param) * n)
        C = np.matrix(Rectangular_Real_SRT(n, p, s=list(np.linspace(a, b, pp)) + [0] * (n - pp)))
        TrueTotalCovariance = np.matrix(np.bmat([[np.eye(n), C], [C.T, (C.T) * C + sigmatwo * np.eye(p)]]))
    elif model == 'wishart':
        TrueTotalCovariance = Wishart(n=n + p, p=int(param) * (n + p), real=True)
    elif model == 'heavy_tailed':
        TrueTotalCovariance = Heavy_Tailed_Empirical_Covariance_Matrix(n=n + p, p=2 * (n + p), alpha=float(param))
    elif model == 'mixed_wishart':
        CXY = get_submatrices_of_full_cov_mat(n=n,
                                              p=p,
                                              CZZ=Wishart(n=n + p, p=int(param) * (n + p), real=True))[2]
        CXY_norm = np.linalg.norm(CXY, ord=2)
        TrueTotalCovariance = np.matrix(np.bmat([[(CXY_norm) * np.eye(n), CXY],
                                                 [CXY.T, (CXY_norm) * np.eye(p)]]))
    else:
        CXY = get_submatrices_of_full_cov_mat(n=n,
                                              p=p,
                                              CZZ=Heavy_Tailed_Empirical_Covariance_Matrix(n=n + p,
                                                                                           p=2 * (n + p),
                                                                                           alpha=float(param)))[2]
        CXY_norm = np.linalg.norm(CXY, ord=2)
        TrueTotalCovariance = np.matrix(np.bmat([[(CXY_norm) * np.eye(n), CXY],
                                                 [CXY.T, (CXY_norm) * np.eye(p)]]))
    C = TrueTotalCovariance[np.ix_(range(n), range(n, n + p))]
    Etotale = Empirical_Covariance(T, TrueTotalCovariance)
    paper_RIE = RIE_Cross_Covariance(Etotale, T, n, p)
    projected_RIE = RIE_Covariance(Etotale, T)[np.ix_(range(n), range(n, n + p))]
    emp_estim = Etotale[np.ix_(range(n), range(n, n + p))]
    res = {}
    for norm in ('frobenius', 'operator_norm'):
        res[norm] = dict(emp=error_norm(emp_estim, C, norm),
                         rie_paper=error_norm(paper_RIE, C, norm),
                         proj_ledoit_peche=error_norm(projected_RIE, C, norm))
    return pd.DataFrame(res)


def Affichage(L, my_name):
    x, y = np.mean(L), np.std(L)
    print("\n\t\t" + my_name + ":")
    print("\t\t%.2e   +/-   %.1e" % (x, 1.96 * y * N_LGN ** (-.5)))


for model_param in tested_cases:
    t0prog = time()
    diff_RIE = []
    diff_Emp = []
    diff_RIE_Total = []
    diff_RIE_operator_norm = []
    diff_Emp_operator_norm = []
    diff_RIE_Total_operator_norm = []
    for i in range(N_LGN):
        res = estimator_comparison(T=T, model_param=model_param)
        diff_Emp.append(res.loc['emp', 'frobenius'])
        diff_RIE.append(res.loc['rie_paper', 'frobenius'])
        diff_RIE_Total.append(res.loc['proj_ledoit_peche', 'frobenius'])
        diff_Emp_operator_norm.append(res.loc['emp', 'operator_norm'])
        diff_RIE_operator_norm.append(res.loc['rie_paper', 'operator_norm'])
        diff_RIE_Total_operator_norm.append(res.loc['proj_ledoit_peche', 'operator_norm'])

    print(f"\n\n {model_param} :")
    print("\n\tFrobenius norm:")
    Affichage(np.array(diff_RIE) / np.array(diff_Emp), "RIE Cross Cov/Emp:")
    Affichage(np.array(diff_RIE) / np.array(diff_RIE_Total), "RIE Cross Cov/RIE Ledoit Peche:")
    print("\n\tOperator norm:")
    Affichage(np.array(diff_RIE_operator_norm) / np.array(diff_Emp_operator_norm), "RIE Cross Cov/Emp:")
    Affichage(np.array(diff_RIE_operator_norm) / np.array(diff_RIE_Total_operator_norm),
              "RIE Cross Cov/RIE Ledoit Peche:")

    Tprog = time() - t0prog
    if Tprog < 60:
        print(f"\n\nTime in seconds for {model_param}: " + str(round(Tprog, 2)))
    else:
        print(f"\n\nTime in minutes for {model_param}: " + str(round(Tprog / 60, 1)))
