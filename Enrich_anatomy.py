import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats
import matplotlib

### Calculating the enrichemnt of anatomical communities for cell roles and
### sensory inter and motor neurons

matplotlib.rcParams['pdf.fonttype']=42


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



levels = state.get_levels()
levels_aconn = state_aconn.get_levels()


groups_heirarchical_aconn = []
for i in np.arange(188):
    groups_heirarchical_aconn.append(levels_aconn[0].get_blocks()[i])

group_lengths_aconn = []
for gr in np.unique(groups_heirarchical_aconn):
    group_lengths_aconn.append(len(np.where(groups_heirarchical_aconn == gr)[0]))
group_sorter_aconn = np.argsort(group_lengths_aconn)

groups_heirarchical_unique_aconn = np.unique(groups_heirarchical_aconn)[group_sorter_aconn]


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

vncmotor = ["DA1", "DB1", "DB2", "AS1", "DD1", "VA1", "VB1", "VB2", "VD1", "VD2"]

locomotoryinterneurons = ["AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER"]



Amphid = ["ADFL", "ADLL", "AFDL", "ASEL", "ASGL", "ASHL", "ASIL", "ASJL", "ASKL", "AWAL", "AWBL", "AWCL", "ADFR",
          "ADLR", "AFDR","ASER", "ASGR", "ASHR", "ASIR", "ASJR", "ASKR", "AWAR", "AWBR", "AWCR"]



pharynx = ["I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6", "M1", "M2L", "M2R",
           "M3L", "M3R", "M4", "M5", "MCL", "MCR", "MI", "NSML", "NSMR"]

ringinterneurons = ["ADAL", "ADAR", "AIML", "AIMR", "AINL", "AINR", "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR",
                    "RID", "RIFL", "RIFR", "RIGL", "RIGR", "RIH", "RIPL", "RIPR", "RIR", "RIS", "RIVL", "RIVR",
                    "SAADL", "SAADR", "SAAVL", "SAAVR", "SABD", "SABVL", "SABVR"]

all_other = []
for neuron in neuronlist:
    if neuron not in ringinterneurons + Amphid + locomotoryinterneurons + pharynx + vncmotor:
        all_other.append(neuron)

#calculating the overlap between motor sensory and inter with the modules
#1 is sensory, 2 is inter, 3 is motor
overlap_SIM = np.zeros((len(groups_heirarchical_unique_aconn),3))
overlap_pharynx = np.zeros((len(groups_heirarchical_unique_aconn),1))
overlap_amphid = np.zeros((len(groups_heirarchical_unique_aconn),1))
overlap_vncmotor = np.zeros((len(groups_heirarchical_unique_aconn),1))
overlap_locomotoryinterneurons = np.zeros((len(groups_heirarchical_unique_aconn),1))
overlap_ringinterneurons = np.zeros((len(groups_heirarchical_unique_aconn),1))
overlap_allother = np.zeros((len(groups_heirarchical_unique_aconn),1))
for i in np.arange(len(groups_heirarchical_unique_aconn)):
    indicies_of_group = np.where(groups_heirarchical_aconn == groups_heirarchical_unique_aconn[i])[0]
    neurons_in_group = np.array(neuronlist)[indicies_of_group]
    for neuron in neurons_in_group:
        if neuron in sensory:
            overlap_SIM[i,0] += 1
        elif neuron in inter:
            overlap_SIM[i, 1] += 1
        elif neuron in motor:
            overlap_SIM[i, 2] += 1

        if neuron in pharynx:
            overlap_pharynx[i] += 1
        if neuron in Amphid:
            overlap_amphid[i] += 1
        if neuron in vncmotor:
            overlap_vncmotor[i] += 1
        if neuron in locomotoryinterneurons:
            overlap_locomotoryinterneurons[i] += 1
        if neuron in ringinterneurons:
            overlap_ringinterneurons[i] += 1
        if neuron in all_other:
            overlap_allother[i] += 1

overlap_SIM_permutations = np.zeros((len(groups_heirarchical_unique_aconn),3,1000))
overlap_pharynx_permutations = np.zeros((len(groups_heirarchical_unique_aconn),1, 1000))
overlap_amphid_permutations = np.zeros((len(groups_heirarchical_unique_aconn),1, 1000))
overlap_locomotoryinterneurons_permutations = np.zeros((len(groups_heirarchical_unique_aconn),1, 1000))
overlap_vncmotor_permutations = np.zeros((len(groups_heirarchical_unique_aconn),1, 1000))
overlap_ringinterneurons_permutations = np.zeros((len(groups_heirarchical_unique_aconn),1, 1000))
overlap_allother_permutations = np.zeros((len(groups_heirarchical_unique_aconn),1, 1000))

for r in np.arange(1000):
    groups_heirarchical_permute = np.random.permutation(groups_heirarchical_aconn)
    for i in np.arange(len(groups_heirarchical_unique_aconn)):
        indicies_of_group = np.where(groups_heirarchical_permute == groups_heirarchical_unique_aconn[i])[0]
        neurons_in_group = np.array(neuronlist)[indicies_of_group]
        for neuron in neurons_in_group:
            if neuron in sensory:
                overlap_SIM_permutations[i, 0, r] += 1
            elif neuron in inter:
                overlap_SIM_permutations[i, 1, r] += 1
            elif neuron in motor:
                overlap_SIM_permutations[i, 2, r] += 1
            if neuron in pharynx:
                overlap_pharynx_permutations[i, 0, r] += 1
            if neuron in Amphid:
                overlap_amphid_permutations[i, 0, r] += 1
            if neuron in locomotoryinterneurons:
                overlap_locomotoryinterneurons_permutations[i, 0, r] += 1
            if neuron in vncmotor:
                overlap_vncmotor_permutations[i, 0, r] += 1
            if neuron in pharynx:
                overlap_pharynx_permutations[i, 0, r] += 1
            if neuron in ringinterneurons:
                overlap_ringinterneurons_permutations[i, 0, r] += 1
            if neuron in all_other:
                overlap_allother_permutations[i, 0, r] += 1



overlap_SIM_permutations_mean = np.mean(overlap_SIM_permutations, axis = 2)
overlap_SIM_permutations_std = np.std(overlap_SIM_permutations, axis = 2)

overlap_pharynx_permutations_mean = np.mean(overlap_pharynx_permutations, axis = 2)
overlap_pharynx_permutations_std = np.std(overlap_pharynx_permutations, axis = 2)

overlap_SIM_zscore = (overlap_SIM - overlap_SIM_permutations_mean)/overlap_SIM_permutations_std
overlap_SIM_pvalues = np.zeros((len(np.unique(groups_heirarchical_aconn)),3))

overlap_pharynx_zscore = (overlap_pharynx - overlap_pharynx_permutations_mean)/overlap_pharynx_permutations_std
overlap_pharynx_pvalues = np.zeros((len(np.unique(groups_heirarchical_aconn)),1))

overlap_amphid_permutations_mean = np.mean(overlap_amphid_permutations, axis = 2)
overlap_amphid_permutations_std = np.std(overlap_amphid_permutations, axis = 2)

overlap_locomotoryinterneurons_permutations_mean = np.mean(overlap_locomotoryinterneurons_permutations, axis = 2)
overlap_locomotoryinterneurons_permutations_std = np.std(overlap_locomotoryinterneurons_permutations, axis = 2)
overlap_vncmotor_permutations_mean = np.mean(overlap_vncmotor_permutations, axis = 2)
overlap_vncmotor_permutations_std = np.std(overlap_vncmotor_permutations, axis = 2)

overlap_ringinterneurons_permutations_mean = np.mean(overlap_ringinterneurons_permutations, axis = 2)
overlap_ringinterneurons_permutations_std = np.std(overlap_ringinterneurons_permutations, axis = 2)

overlap_allother_permutations_mean = np.mean(overlap_allother_permutations, axis = 2)
overlap_allother_permutations_std = np.std(overlap_allother_permutations, axis = 2)

overlap_amphid_zscore = (overlap_amphid - overlap_amphid_permutations_mean)/overlap_amphid_permutations_std
overlap_amphid_pvalues = np.zeros((len(np.unique(groups_heirarchical_aconn)),1))

overlap_locomotoryinterneurons_zscore = (overlap_locomotoryinterneurons - overlap_locomotoryinterneurons_permutations_mean)/overlap_locomotoryinterneurons_permutations_std
overlap_locomotoryinterneurons_pvalues = np.zeros((len(np.unique(groups_heirarchical_aconn)),1))
overlap_vncmotor_zscore = (overlap_vncmotor - overlap_vncmotor_permutations_mean)/overlap_vncmotor_permutations_std
overlap_vncmotor_pvalues = np.zeros((len(np.unique(groups_heirarchical_aconn)),1))


overlap_ringinterneurons_zscore = (overlap_ringinterneurons - overlap_ringinterneurons_permutations_mean)/overlap_ringinterneurons_permutations_std
overlap_ringinterneurons_pvalues = np.zeros((len(np.unique(groups_heirarchical_aconn)),1))

overlap_allother_zscore = (overlap_allother - overlap_allother_permutations_mean)/overlap_allother_permutations_std
overlap_allother_pvalues = np.zeros((len(np.unique(groups_heirarchical_aconn)),1))


for b in np.arange(len(groups_heirarchical_unique_aconn)):
    for sim in np.arange(3):
        overlap_SIM_pvalues[b,sim] = stats.ttest_ind(overlap_SIM[b,sim], overlap_SIM_permutations[b,sim,:], alternative = "greater").pvalue
    overlap_pharynx_pvalues[b] = stats.ttest_ind(overlap_pharynx[b], overlap_pharynx_permutations[b, 0, :],
                                                  alternative="greater").pvalue
    overlap_amphid_pvalues[b] = stats.ttest_ind(overlap_amphid[b], overlap_amphid_permutations[b, 0, :],
                                                  alternative="greater").pvalue
    overlap_locomotoryinterneurons_pvalues[b] = stats.ttest_ind(overlap_locomotoryinterneurons[b], overlap_locomotoryinterneurons_permutations[b, 0, :],
                                                alternative="greater").pvalue
    overlap_vncmotor_pvalues[b] = stats.ttest_ind(overlap_vncmotor[b], overlap_vncmotor_permutations[b, 0, :],
                                                    alternative="greater").pvalue
    overlap_ringinterneurons_pvalues[b] = stats.ttest_ind(overlap_ringinterneurons[b], overlap_ringinterneurons_permutations[b, 0, :],
                                                 alternative="greater").pvalue
    overlap_allother_pvalues[b] = stats.ttest_ind(overlap_allother[b], overlap_allother_permutations[b, 0, :],
                                                  alternative="greater").pvalue

overlap_SIM_pvalues_flattened = overlap_SIM_pvalues.flatten()
overlap_SIM_pvalues_corrected = stats.false_discovery_control(overlap_SIM_pvalues_flattened)
overlap_SIM_pvalues_corrected_matrix = overlap_SIM_pvalues_corrected.reshape(len(groups_heirarchical_unique_aconn),3)

overlap_pharynx_pvalues_flattened = overlap_pharynx_pvalues.flatten()
overlap_pharynx_pvalues_corrected = stats.false_discovery_control(overlap_pharynx_pvalues_flattened)
overlap_pharynx_pvalues_corrected_matrix = overlap_pharynx_pvalues_corrected.reshape(len(groups_heirarchical_unique_aconn),1)

overlap_amphid_pvalues_flattened = overlap_amphid_pvalues.flatten()
overlap_amphid_pvalues_corrected = stats.false_discovery_control(overlap_amphid_pvalues_flattened)
overlap_amphid_pvalues_corrected_matrix = overlap_amphid_pvalues_corrected.reshape(len(groups_heirarchical_unique_aconn),1)

overlap_locomotoryinterneurons_pvalues_flattened = overlap_locomotoryinterneurons_pvalues.flatten()
overlap_locomotoryinterneurons_pvalues_corrected = stats.false_discovery_control(overlap_locomotoryinterneurons_pvalues_flattened)
overlap_locomotoryinterneurons_pvalues_corrected_matrix = overlap_locomotoryinterneurons_pvalues_corrected.reshape(len(groups_heirarchical_unique_aconn),1)

overlap_vncmotor_pvalues_flattened = overlap_vncmotor_pvalues.flatten()
overlap_vncmotor_pvalues_corrected = stats.false_discovery_control(overlap_vncmotor_pvalues_flattened)
overlap_vncmotor_pvalues_corrected_matrix = overlap_vncmotor_pvalues_corrected.reshape(len(groups_heirarchical_unique_aconn),1)


overlap_pharynx_pvalues_flattened = overlap_pharynx_pvalues.flatten()
overlap_pharynx_pvalues_corrected = stats.false_discovery_control(overlap_pharynx_pvalues_flattened)
overlap_pharynx_pvalues_corrected_matrix = overlap_pharynx_pvalues_corrected.reshape(len(groups_heirarchical_unique_aconn),1)

overlap_ringinterneurons_pvalues_flattened = overlap_ringinterneurons_pvalues.flatten()
overlap_ringinterneurons_pvalues_corrected = stats.false_discovery_control(overlap_ringinterneurons_pvalues_flattened)
overlap_ringinterneurons_pvalues_corrected_matrix = overlap_ringinterneurons_pvalues_corrected.reshape(len(groups_heirarchical_unique_aconn),1)


overlap_allother_pvalues_flattened = overlap_allother_pvalues.flatten()
overlap_allother_pvalues_corrected = stats.false_discovery_control(overlap_allother_pvalues_flattened)
overlap_allother_pvalues_corrected_matrix = overlap_allother_pvalues_corrected.reshape(len(groups_heirarchical_unique_aconn),1)


#create a matrix with all the categories, i should have done it like this to begin with but oh well
# 0: pharynx, 1: amphids, 2: locomotory interneurons, 3: vnc motor neurons
overlap_all = np.zeros((len(groups_heirarchical_unique_aconn),6))
overlap_all_zscore = np.zeros((len(groups_heirarchical_unique_aconn),6))
overlap_all_pvalues_corrected_matrix = np.zeros((len(groups_heirarchical_unique_aconn),6))

overlap_all[:,0] = overlap_pharynx[:,0]
overlap_all[:,1] = overlap_amphid[:,0]
overlap_all[:,2] = overlap_locomotoryinterneurons[:,0]
overlap_all[:,3] = overlap_vncmotor[:,0]
overlap_all[:,4] = overlap_ringinterneurons[:,0]
overlap_all[:,5] = overlap_allother[:,0]

overlap_all_zscore[:,0] = overlap_pharynx_zscore[:,0]
overlap_all_zscore[:,1] = overlap_amphid_zscore[:,0]
overlap_all_zscore[:,2] = overlap_locomotoryinterneurons_zscore[:,0]
overlap_all_zscore[:,3] = overlap_vncmotor_zscore[:,0]
overlap_all_zscore[:,4] = overlap_ringinterneurons_zscore[:,0]
overlap_all_zscore[:,5] = overlap_allother_zscore[:,0]

overlap_all_pvalues_corrected_matrix[:,0] = overlap_pharynx_pvalues_corrected_matrix[:,0]
overlap_all_pvalues_corrected_matrix[:,1] = overlap_amphid_pvalues_corrected_matrix[:,0]
overlap_all_pvalues_corrected_matrix[:,2] = overlap_locomotoryinterneurons_pvalues_corrected_matrix[:,0]
overlap_all_pvalues_corrected_matrix[:,3] = overlap_vncmotor_pvalues_corrected_matrix[:,0]
overlap_all_pvalues_corrected_matrix[:,4] = overlap_ringinterneurons_pvalues_corrected_matrix[:,0]
overlap_all_pvalues_corrected_matrix[:,5] = overlap_allother_pvalues_corrected_matrix[:,0]


fig, ax = plt.subplots()
plt.imshow(overlap_all_zscore, cmap="coolwarm", vmin = -4, vmax = 4)
for i in np.arange(len(groups_heirarchical_unique_aconn)):
    for j in np.arange(6):
        if overlap_all_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(j, i, "**", va='center', ha='center')
        elif overlap_all_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(j, i, "*", va='center', ha='center')
plt.xticks(np.arange(6), ["Pharyngeal neurons", "Amphid sensory neurons", "Locomotory interneurons",
                          "VNC motorneurons", "Ring interneurons", "All others"])
plt.yticks(np.arange(20), np.arange(20)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/neurontype_enrichment_Aconn_zscores_mcmc_other.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig2/neurontype_enrichment_Aconn_zscores_mcmc_other.eps",
    dpi=300, bbox_inches="tight", format="eps")
plt.show()

fig, ax = plt.subplots()
plt.imshow(overlap_SIM_zscore, cmap="coolwarm", vmin = -4, vmax = 4)
for i in np.arange(len(groups_heirarchical_unique_aconn)):
    for j in np.arange(3):
        if overlap_SIM_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(j, i, "**", va='center', ha='center')
        elif overlap_SIM_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(j, i, "*", va='center', ha='center')

plt.xticks(np.arange(3), ["Sensoryneuron", "Interneuron", "Motorneuron"])
plt.yticks(np.arange(20), np.arange(20)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)

plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment_Aconn_zscores_mcmc.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig2/SIM_enrichment_Aconn_zscores_mcmc.eps",
    dpi=300, bbox_inches="tight", format ="eps")
plt.show()

print("done")