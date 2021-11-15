"""
Created on Sun Dec 16 17:09:00 2018

@author: florent
"""

from base_functions import *

Tis, Tos = 1000, 100
test_run = 0
if not test_run:
    f = 10
    Tis *= f
    Tos *= f

a, b = 0.6, 0.8
alpha, beta = 0.1, 0.1
proportion_info = 1
eps = 0.01
invariance_flag = 1
figsize = (5, 4)

n, p = my_int(alpha * Tis), my_int(beta * Tis)


def my_local_model(proportion_info, invariance_flag=1):
    size_info = my_int(proportion_info * n)
    s = [0] * (n - size_info) + list(np.linspace(a, b, size_info))
    return (
        (sorted(s, reverse=1), Rectangular_Real_SRT(n, p, s))
        if invariance_flag
        else (sorted(s, reverse=1), np.bmat([np.diag(s), np.zeros((n, p - n))]))
    )


(s_True, A) = my_local_model(proportion_info, invariance_flag)
if proportion_info > 0:
    c = np.sqrt(n) / np.linalg.norm(A)
    A *= c
    s_True = np.array(s_True) * c


def data_set(T):
    F = np.matrix(np.random.randn(p, T))
    R = A * F + eps * np.random.randn(n, T)
    return (F, R)


Fis, Ris = data_set(Tis)
Fos, Ros = data_set(Tos)

Zis = np.matrix(np.bmat([[Ris], [Fis]]))
Etoteis = Zis * (Zis.T) / Tis
CRFos = np.matrix(Ros) * np.matrix(Fos.T) / Tos

s, U, V, new_s, new_s_isotonic = RIE_Cross_Covariance(CZZemp=Etoteis,
                                                      T=Tis, n=n, p=p,
                                                      return_all=1)

Oracle_os = (U.T) * CRFos * (V[:n, :].T)

True_Oracle = (U.T) * A * (V[:n, :].T)

os_overlap_over_is_overlap = []
s_cleaned_over_s = []
s_cleaned_over_s_isotonic = []
True_overlap_over_is_overlap = []

for k in range(n):
    True_overlap_over_is_overlap.append(True_Oracle[k, k] / (s[k]))
    os_overlap_over_is_overlap.append(Oracle_os[k, k] / (s[k]))
    s_cleaned_over_s.append(new_s[k] / (s[k]))
    s_cleaned_over_s_isotonic.append(new_s_isotonic[k] / (s[k]))

dimension_details = f"Tis={Tis}_Toos={Tos}_n={n}_p={p}"
outdir = "data_and_figures/"
outdir += sys.argv[0].split("/")[-1].replace(".py", "")
outdir += f"/figures_{dimension_details}/"
os.makedirs(outdir, exist_ok=True)

fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
ax.plot(s_True, s, "k.", label="Empirical")
ax.plot(s_True, new_s, "b.", label="Cleaned")
ax.plot(s_True, new_s_isotonic, "g.", label="Cleaned Iso")
ax.plot(s_True, s_True, "r.", label="True")
# plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel("True singular values")  # , fontsize=15)
plt.legend(loc="best")
plt.ylim(bottom=0)
plt.xlabel("True singular value")
title = "True vs emp and cleaned singular values"
plt.title(title)
file_name = os.path.join(outdir, (f"{title}_{dimension_details}.png").replace(" ", "_"))
plt.savefig(file_name, bbox_inches="tight")

fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
ax.plot(range(n), s[::-1], "r", label="Empirical")
ax.plot(range(n), new_s[::-1], "b", label="Cleaned")
# plt.plot(range(n),new_s_isotonic,"g",label="Cleaned_iso")
ax.plot(range(n), s_True[::-1], "m", label="True")
plt.legend(loc="best")
title = "Singular values"
plt.title(title)
plt.xlabel("Singular value index (increasingly ordered)")
file_name = os.path.join(outdir, (f"{title}_{dimension_details}.png").replace(" ", "_"))
plt.savefig(file_name, bbox_inches="tight")

fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
ax.plot(range(n), s_cleaned_over_s[::-1], "b", label="Cleaned/Empirical")
ax.plot(
    range(n), s_cleaned_over_s_isotonic[::-1], "m", label="Cleaned isotonic/Empirical"
)
# plt.ylim(bottom=0,top=1)
plt.legend(loc="best")
title = "Cleaning quotients"
plt.title(title)
plt.xlabel("Singular value index (increasingly ordered)")
file_name = os.path.join(outdir, (f"{title}_{dimension_details}.png").replace(" ", "_"))
plt.savefig(file_name, bbox_inches="tight")

fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
ax.plot(range(n), True_overlap_over_is_overlap[::-1], "g", label="True ovl/IS ovl")
ax.plot(range(n), os_overlap_over_is_overlap[::-1], "r", label="OOS ovl/IS ovl")
ax.plot(range(n), s_cleaned_over_s[::-1], "b", label="Cleaned/Empirical")
# plt.plot(range(n),s_cleaned_over_s_isotonic,"m",label="Cleaned_iso/Empirical")
plt.ylim(bottom=0)
plt.legend(loc="best")
title = "Overlaps quotients"
plt.title(title)
plt.xlabel("Singular value index (increasingly ordered)")
file_name = os.path.join(outdir, (f"{title}_{dimension_details}.png").replace(" ", "_"))
plt.savefig(file_name, bbox_inches="tight")

fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
ax.plot(
    range(n),
    os_overlap_over_is_overlap[::-1],
    "r",
    linewidth=1,
    label="$\mathrm{OVL}_{oos}/\mathrm{OVL}_{is}$",
)
ax.plot(
    range(n),
    s_cleaned_over_s_isotonic[::-1],
    "b",
    linewidth=1,
    label="$s_k^{\mathrm{cleaned}}/s_k$",
)
plt.axhline(y=1, color="k", linestyle="--", label="$y=1$")
plt.tick_params(axis="both", which="major", labelsize=15)
plt.ylim(bottom=0)
plt.legend(loc="best")
title = "Overfitting factor and cleaning"
# plt.title(title, fontsize=20)
plt.xlabel("Singular value index $k$ (increasingly ordered)")
file_name = os.path.join(outdir, (f"{title}_{dimension_details}.png").replace(" ", "_"))
plt.savefig(file_name, bbox_inches="tight")

# plt.figure()
# plt.plot(range(n),os_overlap_over_is_overlap,"r")
# plt.plot(range(n),s_cleaned_over_s_isotonic,"b")
