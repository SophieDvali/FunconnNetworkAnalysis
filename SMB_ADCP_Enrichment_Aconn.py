import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import sys
import matplotlib

matplotlib.rcParams['pdf.fonttype']=42


### Calculating the enrichement of different neurons for participation in community motifs in anatomy


# Ci gives the block that each of the nodes is in
df_Ci = pd.read_csv(
    '/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Ci_group_values_aconn.csv', header=None)
Ci = df_Ci.to_numpy()
# Ci = np.delete(Ci, obj=0, axis=1)
# Morph: the first 3 columns are w_rr, w_ss, w_sr, the 4th column is the type of community relationship (1, asso, 2: diss, 3: core-peri)
# 5th and 6th columns are r and s
df_Morph = pd.read_csv(
    '/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Morph_values_aconn.csv', header=None)
Morph = df_Morph.to_numpy()
# Morph = np.delete(Morph, obj=0, axis=1)
# C gives the community properties of individual nodes, first 4 columns are asso, diss, core and periphery
df_C_values = pd.read_csv(
    '/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/C_values_aconn.csv',
    header=None)
C_values = df_C_values.to_numpy()
# C_values = np.delete(C_values, obj=0, axis=1)


with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/get_adj_matrix.pkl', 'rb') as f:
    get_adj_matrix = pickle.load(f)
with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/get_adj_matrix_aconn.pkl', 'rb') as f:
    get_adj_matrix_aconn = pickle.load(f)
with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/neuronlist.pkl', 'rb') as g:
    neuronlist = pickle.load(g)


#make graphs
g = gt.Graph(get_adj_matrix)
g_aconn = gt.Graph(get_adj_matrix_aconn)

#define vector and edge properites
vprop = g.new_vertex_property("string")
g.vp.neuron = vprop
vprop2 = g.new_vertex_property("int")
g.vp.number = vprop2

n = 0
for v in g.vertices():
    g.vp.neuron[v] = neuronlist[n]
    g.vp.number[v] = n
    n+=1

vprop_aconn = g_aconn.new_vertex_property("string")
g_aconn.vp.neuron = vprop_aconn
n = 0
for va in g_aconn.vertices():
    g_aconn.vp.neuron[va] = neuronlist[n]
    n += 1

with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/fconn_state_mcmc.pickle',
          'rb') as st:
    state = pickle.load(st)
with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/aconn_state_mcmc.pickle',
          'rb') as st_a:
    state_aconn = pickle.load(st_a)


levels_aconn = state_aconn.get_levels()


groups_heirarchical_aconn = []
for i in np.arange(188):
    groups_heirarchical_aconn.append(levels_aconn[0].get_blocks()[i])

group_lengths = []
for gr in np.unique(groups_heirarchical_aconn):
    group_lengths.append(len(np.where(groups_heirarchical_aconn == gr)[0]))

group_sorter = np.argsort(group_lengths)

# the order of the groups is determined by this so that they are always the same
groups_heirarchical_unique = np.unique(groups_heirarchical_aconn)[group_sorter]


#morph: wrr, wss, wsr, type(1:asso, 2: diss, 3:core-periphery), r, s
relationships = np.zeros((len(groups_heirarchical_unique), len(groups_heirarchical_unique)))

for m in np.arange(len(Morph)):
    r = int(Morph[m,4])-1
    s = int(Morph[m, 5])-1
    relationships[r,s] = Morph[m, 3]

frac_asso = len(np.where(Morph[:,3] == 1)[0])/(len(Morph[:,3]))
frac_cp = len(np.where(Morph[:,3] == 3)[0])/(len(Morph[:,3]))

sensory = ['ADEL', 'ADER', 'ADFL', 'ADFR', 'ADLL', 'ADLR', 'AFDL', 'AFDR',
           'AQR', 'ASEL', 'ASER', 'ASGL', 'ASGR', 'ASHL', 'ASHR', 'ASIL',
           'ASIR', 'ASJL', 'ASJR', 'ASKL', 'ASKR', 'AUAL', 'AUAR', 'AVG',
           'AWAL', 'AWAR', 'AWBL', 'AWBR', 'AWCL', 'AWCR', 'BAGL', 'BAGR',
           'CEPDL', 'CEPDR', 'CEPVL', 'CEPVR', 'FLPL', 'FLPR', 'IL1DL',
           'IL1DR', 'IL1L', 'IL1R', 'IL1VL', 'IL1VR', 'IL2DL', 'IL2DR',
           'IL2L', 'IL2R', 'IL2VL', 'IL2VR', 'NSML', 'NSMR', 'OLLL', 'OLLR',
           'OLQDL', 'OLQDR', 'OLQVL', 'OLQVR', 'URADL', 'URADR', 'URAVL',
           'URAVR', 'URBL', 'URBR', 'URXL', 'URXR', 'URYDL', 'URYDR', 'URYVL',
           'URYVR']
inter = ['ADAL', 'ADAR', 'AIAL', 'AIAR', 'AIBL', 'AIBR', 'AIML',
         'AIMR', 'AINL', 'AINR', 'AIYL', 'AIYR', 'AIZL', 'AIZR', 'ALA',
         'AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVEL', 'AVER',
         'AVFL', 'AVFR', 'AVHL', 'AVHR', 'AVJL', 'AVJR', 'AVKL', 'AVKR',
         'AVL', 'I1L', 'I1R', 'I2L', 'I2R', 'I3', 'I4', 'I5', 'I6', 'RIAL',
         'RIAR', 'RIBL', 'RIBR', 'RICL', 'RICR', 'RIFL', 'RIFR', 'RIGL',
         'RIGR', 'RIH', 'RIPL', 'RIPR', 'RIR', 'RIS', 'RIVL', 'RIVR',
         'SAADL', 'SAADR', 'SAAVL', 'SAAVR', 'SABD', 'SABVL', 'SABVR',
         'SIADL', 'SIADR', 'SIAVL', 'SIAVR', 'SIBDL', 'SIBDR', 'SIBVL',
         'SIBVR']
motor = ['AS1', 'DA1', 'DB1', 'DB2', 'DD1', 'M1', 'M2L', 'M2R',
         'M3L', 'M3R', 'M4', 'M5', 'MCL', 'MCR', 'MI', 'RID', 'RIML',
         'RIMR', 'RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR', 'RMED',
         'RMEL', 'RMER', 'RMEV', 'RMFL', 'RMFR', 'RMGL', 'RMGR', 'RMHL',
         'SMBDL', 'SMBDR', 'SMBVL', 'SMBVR', 'SMDDL', 'SMDDR', 'SMDVL',
         'SMDVR', 'VA1', 'VB1', 'VB2', 'VD1', 'VD2']

pharynx = ["I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6", "M1", "M2L", "M2R",
           "M3L", "M3R", "M4", "M5", "MCL", "MCR", "MI", "NSML", "NSMR"]

vncmotor = ["DA1", "DB1", "DB2", "AS1", "DD1", "VA1", "VB1", "VB2", "VD1", "VD2"]

locomotoryinterneurons = ["AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER"]

Amphid = ["ADFL", "ADLL", "AFDL", "ASEL", "ASGL", "ASHL", "ASIL", "ASJL", "ASKL", "AWAL", "AWBL", "AWCL", "ADFR",
          "ADLR", "AFDR","ASER", "ASGR", "ASHR", "ASIR", "ASJR", "ASKR", "AWAR", "AWBR", "AWCR"]

ringinterneurons = ["ADAL", "ADAR", "AIML", "AIMR", "AINL", "AINR", "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR",
                    "RID", "RIFL", "RIFR", "RIGL", "RIGR", "RIH", "RIPL", "RIPR", "RIR", "RIS", "RIVL", "RIVR",
                    "SAADL", "SAADR", "SAAVL", "SAAVR", "SABD", "SABVL", "SABVR"]

richclub_aconn= ['AIBL', 'AIBR', 'AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVEL', 'AVER',
       'RIBL', 'RIBR', 'RIGL', 'RIH', 'RIML', 'RIMR', 'RIS']
richclub_sp = ['ASHL', 'AVDL', 'AVEL', 'AVER', 'AVJL', 'AVJR', 'AWBL', 'AWBR',
       'M3L', 'M3R', 'OLLR', 'OLQDR', 'RMDDL', 'RMDDR', 'RMDVL', 'RMDVR']

all_other = []
for neuron in neuronlist:
    if neuron not in ringinterneurons + Amphid + locomotoryinterneurons + pharynx + vncmotor:
        all_other.append(neuron)

#calculating the ACDP score for each category
inter_inds = []
sensory_inds = []
motor_inds = []
pharynx_inds = []
amphid_inds = []
vncmotor_inds = []
locomotoryinterneurons_inds = []
ringinterneurons_inds = []
richclub_sp_inds = []
richclub_aconn_inds = []
allother_inds = []

for neu in np.arange(len(Ci)):
    neuron = neuronlist[neu]
    if neuron in sensory:
        sensory_inds.append(neu)
    elif neuron in inter:
        inter_inds.append(neu)
    elif neuron in motor:
        motor_inds.append(neu)

    if neuron in pharynx:
        pharynx_inds.append(neu)
    if neuron in Amphid:
        amphid_inds.append(neu)
    if neuron in vncmotor:
        vncmotor_inds.append(neu)
    if neuron in locomotoryinterneurons:
        locomotoryinterneurons_inds.append(neu)
    if neuron in ringinterneurons:
        ringinterneurons_inds.append(neu)
    if neuron in richclub_sp:
        richclub_sp_inds.append(neu)
    if neuron in richclub_aconn:
        richclub_aconn_inds.append(neu)
    if neuron in all_other:
        allother_inds.append(neu)

C_inter = C_values[inter_inds,:]
C_sensory = C_values[sensory_inds,:]
C_motor = C_values[motor_inds,:]
C_pharynx = C_values[pharynx_inds,:]
C_amphid = C_values[amphid_inds,:]
C_vncmotor = C_values[vncmotor_inds,:]
C_locomotoryinterneurons = C_values[locomotoryinterneurons_inds,:]
C_ringinterneurons = C_values[ringinterneurons_inds,:]

C_richclub_sp = C_values[richclub_sp_inds,:]
C_richclub_aconn = C_values[richclub_aconn_inds,:]

C_allother = C_values[allother_inds,:]
###

C_inter_mean = np.mean(C_inter, axis = 0)
C_sensory_mean = np.mean(C_sensory, axis = 0)
C_motor_mean = np.mean(C_motor, axis = 0)
C_pharynx_mean = np.mean(C_pharynx, axis = 0)
C_amphid_mean = np.mean(C_amphid, axis = 0)
C_vncmotor_mean = np.mean(C_vncmotor, axis = 0)
C_locomotoryinterneurons_mean = np.mean(C_locomotoryinterneurons, axis = 0)
C_ringinterneurons_mean = np.mean(C_ringinterneurons, axis = 0)

C_richclub_sp_mean = np.mean(C_richclub_sp, axis = 0)
C_richclub_aconn_mean = np.mean(C_richclub_aconn, axis = 0)

C_allother_mean = np.mean(C_allother, axis = 0)

npermute = 1000

C_inter_mean_permute = np.zeros((len(C_inter_mean), npermute))
C_sensory_mean_permute = np.zeros((len(C_inter_mean), npermute))
C_motor_mean_permute = np.zeros((len(C_inter_mean), npermute))
C_pharynx_mean_permute = np.zeros((len(C_inter_mean), npermute))
C_amphid_mean_permute = np.zeros((len(C_amphid_mean), npermute))
C_vncmotor_mean_permute = np.zeros((len(C_vncmotor_mean), npermute))
C_locomotoryinterneurons_mean_permute = np.zeros((len(C_locomotoryinterneurons_mean), npermute))
C_ringinterneurons_mean_permute = np.zeros((len(C_ringinterneurons_mean), npermute))

C_richclub_sp_mean_permute = np.zeros((len(C_richclub_sp_mean), npermute))
C_richclub_aconn_mean_permute = np.zeros((len(C_richclub_aconn_mean), npermute))

C_allother_mean_permute = np.zeros((len(C_allother_mean), npermute))

#1000 permutations
for r in np.arange(npermute):
    #random permutations of the scores need to be permuted TOGETHER!
    permutation = np.random.permutation(np.arange(len(Ci)))
    C_values_permute = C_values[permutation,:]

    C_inter_permute = C_values_permute[inter_inds, :]
    C_sensory_permute = C_values_permute[sensory_inds, :]
    C_motor_permute = C_values_permute[motor_inds, :]
    C_pharynx_permute = C_values_permute[pharynx_inds, :]
    C_amphid_permute = C_values_permute[amphid_inds, :]
    C_vncmotor_permute = C_values_permute[vncmotor_inds, :]
    C_locomotoryinterneurons_permute = C_values_permute[locomotoryinterneurons_inds, :]
    C_ringinterneurons_permute = C_values_permute[ringinterneurons_inds, :]

    C_richclub_sp_permute = C_values_permute[richclub_sp_inds, :]
    C_richclub_aconn_permute = C_values_permute[richclub_aconn_inds, :]

    C_allother_permute = C_values_permute[allother_inds, :]

    C_inter_mean_permute[:, r] = np.mean(C_inter_permute, axis=0)
    C_sensory_mean_permute[:, r] = np.mean(C_sensory_permute, axis=0)
    C_motor_mean_permute[:, r] = np.mean(C_motor_permute, axis=0)
    C_pharynx_mean_permute[:, r] = np.mean(C_pharynx_permute, axis=0)
    C_amphid_mean_permute[:, r] = np.mean(C_amphid_permute, axis=0)
    C_vncmotor_mean_permute[:, r] = np.mean(C_vncmotor_permute, axis=0)
    C_locomotoryinterneurons_mean_permute[:, r] = np.mean(C_locomotoryinterneurons_permute, axis=0)
    C_ringinterneurons_mean_permute[:, r] = np.mean(C_ringinterneurons_permute, axis=0)

    C_richclub_sp_mean_permute[:, r] = np.mean(C_richclub_sp_permute, axis=0)
    C_richclub_aconn_mean_permute[:, r] = np.mean(C_richclub_aconn_permute, axis=0)

    C_allother_mean_permute[:, r] = np.mean(C_allother_permute, axis=0)

#1 is sensory, 2 is inter, 3 is motor
C_SIM = np.array([C_sensory_mean,C_inter_mean,C_motor_mean])
C_SIM = C_SIM[:,:4]


C_function = np.array([C_pharynx_mean, C_amphid_mean, C_locomotoryinterneurons_mean, C_vncmotor_mean, C_ringinterneurons_mean, C_allother_mean])
C_function = C_function[:,:4]

C_richclub_sp = C_richclub_sp_mean[:4]
C_richclub_aconn = C_richclub_aconn_mean[:4]

C_SIM_permute = np.array([C_sensory_mean_permute,C_inter_mean_permute,C_motor_mean_permute])
C_SIM_permute = C_SIM_permute[:,:4, :]

C_function_permute = np.array([C_pharynx_mean_permute, C_amphid_mean_permute, C_locomotoryinterneurons_mean_permute,
                               C_vncmotor_mean_permute, C_ringinterneurons_mean_permute, C_allother_mean_permute])
C_function_permute = C_function_permute[:,:4]


C_richclub_sp_permute = C_richclub_sp_mean_permute[:4, :]

C_function_permute_mean = np.mean(C_function_permute, axis=2)
C_function_permute_std = np.std(C_function_permute, axis=2)

C_SIM_permute_mean = np.mean(C_SIM_permute, axis=2)
C_SIM_permute_std = np.std(C_SIM_permute, axis=2)

C_richclub_sp_permute_mean = np.mean(C_richclub_sp_permute, axis=1)
C_richclub_sp_permute_std = np.std(C_richclub_sp_permute, axis=1)

C_richclub_aconn_permute = C_richclub_aconn_mean_permute[:4, :]

C_richclub_aconn_permute_mean = np.mean(C_richclub_aconn_permute, axis=1)
C_richclub_aconn_permute_std = np.std(C_richclub_aconn_permute, axis=1)


C_SIM_zscore = (C_SIM - C_SIM_permute_mean)/C_SIM_permute_std
C_SIM_pvalues = np.zeros((3,4))


C_function_zscore = (C_function - C_function_permute_mean)/C_function_permute_std
C_function_pvalues = np.zeros((6,4))

C_richclub_sp_zscore = (C_richclub_sp - C_richclub_sp_permute_mean)/C_richclub_sp_permute_std
C_richclub_sp_pvalues = np.zeros((1,4))

C_richclub_aconn_zscore = (C_richclub_aconn - C_richclub_aconn_permute_mean)/C_richclub_aconn_permute_std
C_richclub_aconn_pvalues = np.zeros((1,4))

for adcp in np.arange(4):
    for sim in np.arange(3):
        C_SIM_pvalues[sim,adcp] = stats.ttest_ind(C_SIM[sim,adcp], C_SIM_permute[sim,adcp,:], alternative = "greater").pvalue
    for func in np.arange(6):
        C_function_pvalues[func,adcp] = stats.ttest_ind(C_function[func,adcp], C_function_permute[func,adcp,:], alternative = "greater").pvalue
    C_richclub_sp_pvalues[0][adcp] = stats.ttest_ind(C_richclub_sp[adcp], C_richclub_sp_permute[adcp, :],
                                                 alternative="greater").pvalue
    C_richclub_aconn_pvalues[0][adcp] = stats.ttest_ind(C_richclub_aconn[adcp], C_richclub_aconn_permute[adcp, :],
                                                 alternative="greater").pvalue




#we have diss here but im including this just incase that changes
C_SIM_pvalues_nonNaN = C_SIM_pvalues[~np.isnan(C_SIM_pvalues)]
C_SIM_pvalues_corrected = stats.false_discovery_control(C_SIM_pvalues_nonNaN)
C_SIM_pvalues_corrected_matrix = C_SIM_pvalues_corrected.reshape(3,4)

C_function_pvalues_nonNaN = C_function_pvalues[~np.isnan(C_function_pvalues)]
#C_function_pvalues_flattened = C_function_pvalues.flatten()
C_function_pvalues_corrected = stats.false_discovery_control(C_function_pvalues_nonNaN)
C_function_pvalues_corrected_matrix = C_function_pvalues_corrected.reshape(6,4)


C_richclub_sp_pvalues_nonNaN = C_richclub_sp_pvalues[~np.isnan(C_richclub_sp_pvalues)]
#C_pharynx_pvalues_flattened = C_pharynx_pvalues.flatten()
C_richclub_sp_pvalues_corrected = stats.false_discovery_control(C_richclub_sp_pvalues_nonNaN)
C_richclub_sp_pvalues_corrected_matrix = C_richclub_sp_pvalues_corrected.reshape(4,1)

C_richclub_aconn_pvalues_nonNaN = C_richclub_aconn_pvalues[~np.isnan(C_richclub_aconn_pvalues)]
#C_pharynx_pvalues_flattened = C_pharynx_pvalues.flatten()
C_richclub_aconn_pvalues_corrected = stats.false_discovery_control(C_richclub_aconn_pvalues_nonNaN)
C_richclub_aconn_pvalues_corrected_matrix = C_richclub_aconn_pvalues_corrected.reshape(4,1)



fig, ax = plt.subplots()
plt.imshow(C_richclub_aconn_zscore[~np.isnan(C_richclub_aconn_zscore)].reshape(4,1), cmap="coolwarm", vmin = -2, vmax = 2)
for i in np.arange(3):
        if C_richclub_aconn_pvalues_corrected_matrix[i] < 0.01:
            ax.text(0, i, "**", va='center', ha='center')
        elif C_richclub_aconn_pvalues_corrected_matrix[i] < 0.05:
            ax.text(0, i, "*", va='center', ha='center')
ax.set_xticklabels(["", "Anatomy Rich Club"])
plt.yticks(np.arange(4),["Assortative", "Dissassortative",  "Core", "Periphery"])
#ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)

plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Richclub_ACDP_zscores_mcmc_aconn.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig5/Richclub_ADCP_zscores_mcmc_aconn.eps",
    dpi=300, bbox_inches="tight", format="eps")
plt.show()


fig, ax = plt.subplots()
plt.imshow(C_richclub_sp_zscore[~np.isnan(C_richclub_sp_zscore)].reshape(4,1), cmap="coolwarm", vmin = -2, vmax = 2)
for i in np.arange(3):
        if C_richclub_sp_pvalues_corrected_matrix[i] < 0.01:
            ax.text(0, i, "**", va='center', ha='center')
        elif C_richclub_sp_pvalues_corrected_matrix[i] < 0.05:
            ax.text(0, i, "*", va='center', ha='center')
ax.set_xticklabels(["", "Signal Prop. Rich Club"])
plt.yticks(np.arange(4),["Assortative", "Dissassortative",  "Core", "Periphery"])
#ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SPRichclubinAnatomymodules_ACDP_zscores_mcmc_aconn.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig5/SPRichclubinAnatomymodules_ADCP_zscores_mcmc_aconn.eps",
    dpi=300, bbox_inches="tight", format="eps")
plt.show()


fig, ax = plt.subplots()
plt.imshow(C_SIM_zscore.reshape(4,3), cmap="coolwarm")
for i in np.arange(3):
    for j in np.arange(4):
        if C_SIM_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(j, i, "**", va='center', ha='center')
        elif C_SIM_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(j, i, "*", va='center', ha='center')
ax.set_xticklabels(["", "Sensoryneuron", "", "Interneuron","", "Motorneuron"])
ax.set_yticklabels(["", "Assortative", "","Dissasortative","", "Core", "","Periphery"])
#ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_ACDP_zscores_mcmc_aconn.pdf", dpi=300, bbox_inches="tight")
plt.show()


fig, ax = plt.subplots()
plt.imshow(C_function_zscore[~np.isnan(C_function_zscore)].reshape(6,4), cmap="coolwarm", vmin = -3, vmax = 3)
for i in np.arange(6):
    for j in np.arange(4):
        if C_function_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(j, i, "**", va='center', ha='center')
        elif C_function_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(j, i, "*", va='center', ha='center')
plt.yticks(np.arange(6), ["Pharyngeal neurons", "Amphid sensory neurons", "Locomotory interneurons",
                          "VNC motorneurons", "Ring interneurons", "All others"])
plt.xticks(np.arange(4),["Assortative", "Dissassortative",  "Core", "Periphery"])
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/pharynx_ADCP_zscores_mcmc_aconn_other.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig4/pharynx_ADCP_zscores_mcmc_aconn_other.eps",
        dpi=300, bbox_inches="tight", format="eps")
plt.show()

### transposed version ####
fig, ax = plt.subplots()
plt.imshow(C_function_zscore[~np.isnan(C_function_zscore)].reshape(6,4).transpose(), cmap="coolwarm", vmin = -3, vmax = 3)
for i in np.arange(6):
    for j in np.arange(4):
        if C_function_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(i, j, "**", va='center', ha='center')
        elif C_function_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(i, j, "*", va='center', ha='center')
plt.xticks(np.arange(6), ["Pharyngeal neurons", "Amphid sensory neurons", "Locomotory interneurons",
                          "VNC motorneurons", "Ring interneurons", "All others"])
plt.yticks(np.arange(4),["Assortative", "Dissassortative",  "Core", "Periphery"])
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/pharynx_ADCP_zscores_mcmc_aconn_other.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig4/pharynx_ADCP_zscores_mcmc_aconn_other.eps",
        dpi=300, bbox_inches="tight", format="eps")
plt.show()





fig, ax = plt.subplots()
plt.imshow(C_function_pvalues_corrected_matrix, vmin = 0, vmax = 1, cmap="Purples_r")
for i in np.arange(5):
    for j in np.arange(4):
        if C_function_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(0, i, "**", color="white", va='center', ha='center')
        elif C_function_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(0, i, "*", color="white", va='center', ha='center')
    plt.yticks(np.arange(5), ["Pharyngeal neurons", "Amphid sensory neurons", "Locomotory interneurons",
                              "VNC motorneurons", "Ring interneurons"])
plt.xticks(np.arange(4),["Assortative", "Dissassortative",  "Core", "Periphery"])
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('corrected p-values', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/pharynx_ADCP_pvals_mcmc_aconn.pdf", dpi=300, bbox_inches="tight")
plt.show()




fig, ax = plt.subplots()
plt.imshow(C_SIM_pvalues_corrected_matrix, vmin = 0, vmax = 1,  cmap="Purples_r")
for i in np.arange(3):
    for j in np.arange(4):
        if C_SIM_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(j, i, "**", color="white", va='center', ha='center')
        elif C_SIM_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(j, i, "*", color="white", va='center', ha='center')
ax.set_xticklabels(["", "Sensoryneuron", "", "Interneuron","", "Motorneuron"])
plt.xticks(np.arange(4),["Assortative", "Dissassortative",  "Core", "Periphery"])
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('corrected p-values', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_ADCP_pvals_mcmc_aconn.pdf", dpi=300, bbox_inches="tight")
plt.show()



C_mean = np.mean(C_values, axis = 0)

plt.figure()
plt.pie(C_mean[0:4], labels=["Assortative", "Disassortative", "Core", "Periphery"])
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/ADCP_frac_score_mcmc_aconn.pdf",
            dpi=300, bbox_inches="tight")

plt.show()




print("done")