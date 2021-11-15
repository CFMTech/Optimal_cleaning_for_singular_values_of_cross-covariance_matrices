"""
Created on Thu Aug 16 17:32:18 2018

@author: florent
"""

from base_functions import *

figsize = (5, 4)

C_eta = 1.

file_name = 'L_approx_verif_bi_modal_density_C_eta=' + str(C_eta).replace(".", "point") + '.png'

len_list_T = 17
# len_list_T=2
min_list_T = 200
step_list_T = 50
list_T = min_list_T + step_list_T * np.arange(len_list_T)
N_LGN = 100
mmodel = 10

plot_std = 1
plot_legend = 0
save_f = 1

if mmodel == 3:
    def diff_rel(x, x_0):
        return np.abs(x - x_0)
else:
    def diff_rel(x, x_0):
        return np.abs(x - x_0) / np.abs(x_0)


def Ploting(L_Mean, L_Std, N_LGN, my_color, my_name, std_plot=1):
    plt.plot(list_T, L_Mean, color=my_color, label=my_name, linewidth=3)
    if std_plot:
        Mplus = np.array(L_Mean) + 1.96 * N_LGN ** (-.5) * np.array(L_Std)
        Mminus = np.array(L_Mean) - 1.96 * N_LGN ** (-.5) * np.array(L_Std)
        plt.plot(list_T, Mplus, color=my_color, linestyle="--")
        plt.plot(list_T, Mminus, color=my_color, linestyle="--")


# synthetic_data_true = dict()
synthetic_data_np = dict()
synthetic_data_random = dict()
synthetic_data_svd = dict()
L_values = dict()
for T in list_T:
    print("Generating synthetic data for T =", T)
    (n, p, TrueTotalCovariance) = my_model(T, mmodel)
    # synthetic_data_true[T] =  TrueTotalCovariance
    synthetic_data_np[T] = (n, p)
    synthetic_data_random[T] = {i: Empirical_Covariance(T, TrueTotalCovariance) for i in range(N_LGN)}
    print("Computation of SVD for T =", T)
    synthetic_data_svd[T] = {i: np.linalg.svd(
        synthetic_data_random[T][i][np.ix_(range(n), range(n, n + p))], full_matrices=True)
        for i in range(N_LGN)}
    print("Computation of L for T =", T)
    C = TrueTotalCovariance[np.ix_(range(n), range(n, n + p))]
    Ecomplex, eta = .5, C_eta * (n * p * T) ** (-1 / 12.)
    z = Ecomplex + 1j * eta
    ztwo = z ** 2
    L_values[T] = dict()
    for i in range(N_LGN):
        Etotale = synthetic_data_random[T][i]
        U, s, V = synthetic_data_svd[T][i]
        stwo = s ** 2
        one_over_ztwo_minus_stwo = 1 / (ztwo - stwo)
        Coeff_L = []
        for k in range(n):
            Coeff_L.append((U[:, k].T * C * (V[k, :].T))[0, 0])
        L_values[T][i] = np.dot((one_over_ztwo_minus_stwo * s), Coeff_L) / float(T)

from time import time

print("Computation of approx of L")
Mean_diff_rel_Formule1 = []
Std_diff_rel_Formule1 = []
errors_f1 = []
Mean_diff_rel_Formule2 = []
Std_diff_rel_Formule2 = []
errors_f2 = []
for T in list_T:
    (n, p) = synthetic_data_np[T]

    Ecomplex, eta = .5, C_eta * (n * p * T) ** (-1 / 12.)
    z = Ecomplex + 1j * eta
    ztwo = z ** 2

    diffs_rels_formule1 = []
    diffs_rels_formule2 = []
    diffs_formule1 = []
    diffs_formule2 = []
    for i in range(N_LGN):
        Etotale = synthetic_data_random[T][i]
        CXXemp, CYYemp, CXYemp = get_submatrices_of_full_cov_mat(n=n, p=p, CZZ=Etotale)
        U, s, V = synthetic_data_svd[T][i]
        approx_L = approx_L_or_imLoimH(z=z,
                                       n=n,
                                       p=p,
                                       T=T,
                                       U=U,
                                       s=s,
                                       V=V,
                                       CXXemp=CXXemp,
                                       CYYemp=CYYemp,
                                       CXYemp=CXYemp,
                                       algo_used=1,
                                       return_L=True)
        diffs_rels_formule1.append(diff_rel(approx_L, L_values[T][i]))
        diffs_formule1.append(np.abs(approx_L - L_values[T][i]))
        approx_L = approx_L_or_imLoimH(z=z,
                                       n=n,
                                       p=p,
                                       T=T,
                                       s=s,
                                       algo_used=2,
                                       return_L=True)
        diffs_rels_formule2.append(diff_rel(approx_L, L_values[T][i]))
        diffs_formule2.append(np.abs(approx_L - L_values[T][i]))
    Mean_diff_rel_Formule1.append(np.mean(diffs_rels_formule1))
    Std_diff_rel_Formule1.append(np.std(diffs_rels_formule1))
    Mean_diff_rel_Formule2.append(np.mean(diffs_rels_formule2))
    Std_diff_rel_Formule2.append(np.std(diffs_rels_formule2))
    errors_f1.append(np.mean(diffs_formule1))
    errors_f2.append(np.mean(diffs_formule2))
print("Approx of L computed")

outdir = "data_and_figures/"
outdir += sys.argv[0].split("/")[-1].replace(".py", "")
os.makedirs(outdir, exist_ok=True)

Mean_diff_rel_Formule1 = np.array(Mean_diff_rel_Formule1)
Mean_diff_rel_Formule2 = np.array(Mean_diff_rel_Formule2)

possible_exponents = np.arange(start=0, stop=5.5, step=.001)[1:]

fit_for1 = pd.Series({expo: np.std(errors_f1 * (list_T ** expo)) / np.mean(errors_f1 * (list_T ** expo))
                      for expo in possible_exponents})
print(f"fitted exponent for the error in Formula 1: {fit_for1.idxmin()}")

fit_for2 = pd.Series({expo: np.std(errors_f2 * (list_T ** expo)) / np.mean(errors_f2 * (list_T ** expo))
                      for expo in possible_exponents})
print(f"fitted exponent for the error in Formula 2: {fit_for2.idxmin()}")

plt.figure()
Ploting(Mean_diff_rel_Formule1, Std_diff_rel_Formule1, N_LGN, "r", "Formula 1", std_plot=plot_std)
Ploting(Mean_diff_rel_Formule2, Std_diff_rel_Formule2, N_LGN, "b", "Formula 2", std_plot=plot_std)
plt.ylim(ymin=0)
plt.xlabel("$T$", fontsize=15)
plt.ylabel("Relative differences", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid(True)
if plot_legend:
    plt.legend(loc="best")
if save_f:
    plt.savefig(os.path.join(outdir, file_name), bbox_inches='tight')

print('\nAlgo 1')
time_needed_for_formula1 = []
for T in list_T:
    (n, p) = synthetic_data_np[T]

    t = time()
    for i in range(N_LGN):
        RIE_Cross_Covariance(
            CZZemp=synthetic_data_random[T][i],
            T=T,
            n=n,
            p=p,
            algo_used=1
        )
    time_needed_for_formula1.append(t - time())
print('\nDone for Algo 1')

print('\nAlgo 2')
time_needed_for_formula2 = []
for T in list_T:
    (n, p) = synthetic_data_np[T]

    t = time()
    for i in range(N_LGN):
        RIE_Cross_Covariance(
            CZZemp=synthetic_data_random[T][i],
            T=T,
            n=n,
            p=p,
            algo_used=2
        )
    time_needed_for_formula2.append(t - time())
print('\nDone for Algo 2')

plt.figure()
time_ratios = [float(time_needed_for_formula1[i]) / float(time_needed_for_formula2[i]) for i in range(len(list_T))]
plt.stem(list_T, time_ratios, linewidth=3)
plt.ylim(ymin=0)
plt.xlabel("$T$", fontsize=15)
plt.ylabel("Elapsed time ratio: Algo 1 / Algo 2", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid(True)
if plot_legend:
    plt.legend(loc="best")
if save_f:
    file_name = 'time_f1_over_f2.png'
    plt.savefig(os.path.join(outdir, file_name), bbox_inches='tight')

plt.figure(figsize=(6, 4))
time_ratios = [float(time_needed_for_formula2[i]) / float(time_needed_for_formula1[i]) for i in range(len(list_T))]
plt.stem(list_T, time_ratios, linewidth=3)
plt.ylim(ymin=0)
plt.xlabel("$T$", fontsize=15)
plt.ylabel("Elapsed time ratio: Algo 2 / Algo 1", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid(True)
if plot_legend:
    plt.legend(loc="best")
if save_f:
    file_name = 'time_f2_over_f1.png'
    plt.savefig(os.path.join(outdir, file_name), bbox_inches='tight')
