"""
Created on Thu Aug 16 22:42:48 2018

@author: florent
"""

from base_functions import *

t0prog = time()
a, b = .2, .5
alpha, beta = .4, .7
eps_positive = 1e-3  # t0 make sure we only use positive matrices as covariance matrices
N_LGN = 100
T = 500
sigmatwo = .5
proportion_infos = [0., .1, .2, .3, .4]




def my_model(T):
    n, p = my_int(alpha * T), my_int(beta * T)
    return (n, p, Wishart(n=n+p, p=2*(n+p), real=True))


def diff(x, x_0, ord=None):
    return np.linalg.norm(x - x_0, ord=ord)


def Affichage(L, my_name):
    x, y = np.mean(L), np.std(L)
    print("\n\t\t" + my_name + ":")
    print("\t\t%.2e   +/-   %.1e" % (x, 1.96 * y * N_LGN ** (-.5)))



diff_RIE = []
diff_Emp = []
diff_RIE_Total = []
diff_RIE_operator_norm = []
diff_Emp_operator_norm = []
diff_RIE_Total_operator_norm = []
for i in range(N_LGN):
    (n, p, TrueTotalCovariance) = my_model(T)
    C = TrueTotalCovariance[np.ix_(range(n), range(n, n + p))]
    Etotale = Empirical_Covariance(T, TrueTotalCovariance)
    paper_RIE = RIE_Cross_Covariance(Etotale, T, n, p)
    projected_RIE = RIE_Covariance(Etotale, T)[np.ix_(range(n), range(n, n + p))]
    emp_estim=Etotale[np.ix_(range(n), range(n, n + p))]
    diff_Emp.append(diff(emp_estim, C))
    diff_RIE.append(diff(paper_RIE, C))
    diff_RIE_Total.append(diff(projected_RIE, C))
    diff_Emp_operator_norm.append(diff(emp_estim, C, ord=2))
    diff_RIE_operator_norm.append(diff(paper_RIE, C, ord=2))
    diff_RIE_Total_operator_norm.append(diff(projected_RIE, C, ord=2))
print("\n\n Wishart :")
print("\n\tFrobenius norm:")
Affichage(np.array(diff_RIE) / np.array(diff_Emp), "RIE Cross Cov/Emp:")
Affichage(np.array(diff_RIE) / np.array(diff_RIE_Total), "RIE Cross Cov/RIE Ledoit Peche:")
print("\n\tOperator norm:")
Affichage(np.array(diff_RIE_operator_norm) / np.array(diff_Emp_operator_norm), "RIE Cross Cov/Emp:")
Affichage(np.array(diff_RIE_operator_norm) / np.array(diff_RIE_Total_operator_norm), "RIE Cross Cov/RIE Ledoit Peche:")

Tprog = time() - t0prog
if Tprog < 60:
    print("\n\nTime in seconds : " + str(round(100 * Tprog) / 100.))
else:
    print("\n\nTime in minutes : " + str(round(10 * Tprog / 60) / 10.))
# if Tprog>30:
#     os.system('say "Programme fini"')