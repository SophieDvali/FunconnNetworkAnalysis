
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype']=42


# Ci gives the block that each of the nodes is in
df_Ci_aconn = pd.read_csv(
    '/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Ci_group_values_aconn.csv', header=None)
Ci_aconn = df_Ci_aconn.to_numpy()
# Ci = np.delete(Ci, obj=0, axis=1)
# Morph: the first 3 columns are w_rr, w_ss, w_sr, the 4th column is the type of community relationship (1, asso, 2: diss, 3: core-peri)
# 5th and 6th columns are r and s
df_Morph_aconn = pd.read_csv(
    '/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Morph_values_aconn.csv', header=None)
Morph_aconn = df_Morph_aconn.to_numpy()
# Morph = np.delete(Morph, obj=0, axis=1)
# C gives the community properties of individual nodes, first 4 columns are asso, diss, core and periphery
df_C_values_aconn = pd.read_csv(
    '/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/C_values_aconn.csv',
    header=None)
C_values_aconn = df_C_values_aconn.to_numpy()
# C_values = np.delete(C_values, obj=0, axis=1)

# Ci gives the block that each of the nodes is in
df_Ci = pd.read_csv(
    '/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Ci_group_values_mcmc.csv', header=None)
Ci = df_Ci.to_numpy()
# Ci = np.delete(Ci, obj=0, axis=1)
# Morph: the first 3 columns are w_rr, w_ss, w_sr, the 4th column is the type of community relationship (1, asso, 2: diss, 3: core-peri)
# 5th and 6th columns are r and s
df_Morph = pd.read_csv(
    '/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Morph_values_mcmc.csv', header=None)
Morph = df_Morph.to_numpy()
# Morph = np.delete(Morph, obj=0, axis=1)
# C gives the community properties of individual nodes, first 4 columns are asso, diss, core and periphery
df_C_values = pd.read_csv(
    '/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/C_values_mcmc.csv',
    header=None)
C_values = df_C_values.to_numpy()
# C_values = np.delete(C_values, obj=0, axis=1)

titles = ["Assortative", "Disassortative", "Core", "Periphery"]

fig, axs = plt.subplots(2, 2)
axs = axs.ravel()
for adcp in np.arange(4):
    aconn_cs = C_values_aconn[:,adcp]
    fconn_cs = C_values[:,adcp]
    axs[adcp].plot(aconn_cs, fconn_cs, "o")
    axs[adcp].set_title(titles[adcp])
    axs[adcp].set_xlim([-0.05, 1])
    axs[adcp].set_ylim([-0.05, 1])

#plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
for ax in axs.flat:
    ax.set(xlabel="Score Anatomy", ylabel="Score Signaling")
#    axs[adcp].set_xlabel("Score Anatomy")
 #   axs[adcp].set_xlabel("Score Signaling")
fig.tight_layout()
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/ADCP_compare.pdf",
    dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig4/ADCP_compare.pdf",
    dpi=300, bbox_inches="tight")
plt.show()




print("done")

