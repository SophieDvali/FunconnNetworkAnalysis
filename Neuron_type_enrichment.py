import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats


#import the matricies for the networks and neuronlist
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


# import the precalculated SBM states
with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/fconn_state_mcmc.pickle', 'rb') as st:
    state = pickle.load(st)
with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/aconn_state_mcmc.pickle', 'rb') as st_a:
    state_aconn = pickle.load(st_a)


levels = state.get_levels()


groups_heirarchical = []
for i in np.arange(len(neuronlist)):
    groups_heirarchical.append(levels[0].get_blocks()[i])

#sort communities by the size so that we have a standardized enumeration
group_lengths = []
for gr in np.unique(groups_heirarchical):
    group_lengths.append(len(np.where(groups_heirarchical == gr)[0]))

group_sorter = np.argsort(group_lengths)

groups_heirarchical_unique = np.unique(groups_heirarchical)[group_sorter]


#cell role annotations
vncmotor = ["DA1", "DB1", "DB2", "AS1", "DD1", "VA1", "VB1", "VB2", "VD1", "VD2"]

locomotoryinterneurons = ["AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER"]



Amphid = ["ADFL", "ADLL", "AFDL", "ASEL", "ASGL", "ASHL", "ASIL", "ASJL", "ASKL", "AWAL", "AWBL", "AWCL", "ADFR",
          "ADLR", "AFDR","ASER", "ASGR", "ASHR", "ASIR", "ASJR", "ASKR", "AWAR", "AWBR", "AWCR"]



pharynx = ["I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6", "M1", "M2L", "M2R",
           "M3L", "M3R", "M4", "M5", "MCL", "MCR", "MI", "NSML", "NSMR"]

ringinterneurons = ["ADAL", "ADAR", "AIML", "AIMR", "AINL", "AINR", "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR",
                    "RID", "RIFL", "RIFR", "RIGL", "RIGR", "RIH", "RIPL", "RIPR", "RIR", "RIS", "RIVL", "RIVR",
                    "SAADL", "SAADR", "SAAVL", "SAAVR", "SABD", "SABVL", "SABVR"]

richclub_sp = ['ASHL', 'AVDL', 'AVEL', 'AVER', 'AVJL', 'AVJR', 'AWBL', 'AWBR',
       'M3L', 'M3R', 'OLLR', 'OLQDR', 'RMDDL', 'RMDDR', 'RMDVL', 'RMDVR']



all_other = []
for neuron in neuronlist:
    if neuron not in ringinterneurons + Amphid + locomotoryinterneurons + pharynx + vncmotor:
        all_other.append(neuron)


# finding the overlap between the annotated sets of neurons and the communities
overlap_amphid = np.zeros((len(groups_heirarchical_unique),1))
overlap_vncmotor = np.zeros((len(groups_heirarchical_unique),1))
overlap_locomotoryinterneurons = np.zeros((len(groups_heirarchical_unique),1))
overlap_pharynx = np.zeros((len(groups_heirarchical_unique),1))
overlap_ringinterneurons = np.zeros((len(groups_heirarchical_unique),1))
overlap_allother = np.zeros((len(groups_heirarchical_unique),1))
overlap_rich_club_sp = np.zeros((len(groups_heirarchical_unique),1))
for i in np.arange(len(groups_heirarchical_unique)):
    indicies_of_group = np.where(groups_heirarchical == groups_heirarchical_unique[i])[0]
    neurons_in_group = np.array(neuronlist)[indicies_of_group]
    print("Group:"+str(i+1))
    print(neurons_in_group)
    for neuron in neurons_in_group:
        if neuron in Amphid:
            overlap_amphid[i] += 1
        if neuron in vncmotor:
            overlap_vncmotor[i] += 1
        if neuron in locomotoryinterneurons:
            overlap_locomotoryinterneurons[i] += 1
        if neuron in pharynx:
            overlap_pharynx[i] += 1
        if neuron in ringinterneurons:
            overlap_ringinterneurons[i] += 1
        if neuron in all_other:
            overlap_allother[i] += 1
        if neuron in richclub_sp:
            overlap_rich_club_sp[i] +=1


#shuffling the community assignments to compare to shuffle
overlap_amphid_permutations = np.zeros((len(groups_heirarchical_unique),1, 1000))
overlap_locomotoryinterneurons_permutations = np.zeros((len(groups_heirarchical_unique),1, 1000))
overlap_vncmotor_permutations = np.zeros((len(groups_heirarchical_unique),1, 1000))
overlap_pharynx_permutations = np.zeros((len(groups_heirarchical_unique),1, 1000))
overlap_amphidinter_permutations = np.zeros((len(groups_heirarchical_unique),1, 1000))
overlap_ringinterneurons_permutations = np.zeros((len(groups_heirarchical_unique),1, 1000))
overlap_allother_permutations = np.zeros((len(groups_heirarchical_unique),1, 1000))
overlap_rich_club_sp_permutations = np.zeros((len(groups_heirarchical_unique),1, 1000))
for r in np.arange(1000):
    groups_heirarchical_permute = np.random.permutation(groups_heirarchical)
    for i in np.arange(len(groups_heirarchical_unique)):
        indicies_of_group = np.where(groups_heirarchical_permute == groups_heirarchical_unique[i])[0]
        neurons_in_group = np.array(neuronlist)[indicies_of_group]
        for neuron in neurons_in_group:
            if neuron in Amphid:
                overlap_amphid_permutations[i, 0, r] += 1
            if neuron in locomotoryinterneurons:
                overlap_locomotoryinterneurons_permutations[i, 0, r] += 1
            if neuron in vncmotor:
                overlap_vncmotor_permutations[i, 0, r] += 1
            if neuron in pharynx:
                overlap_pharynx_permutations[i, 0, r] += 1
            if neuron in amphidinter:
                overlap_amphidinter_permutations[i, 0, r] += 1
            if neuron in ringinterneurons:
                overlap_ringinterneurons_permutations[i, 0, r] += 1
            if neuron in all_other:
                overlap_allother_permutations[i, 0, r] += 1
            if neuron in richclub_sp:
                overlap_rich_club_sp_permutations[i, 0, r] += 1


overlap_amphid_permutations_mean = np.mean(overlap_amphid_permutations, axis = 2)
overlap_amphid_permutations_std = np.std(overlap_amphid_permutations, axis = 2)

overlap_locomotoryinterneurons_permutations_mean = np.mean(overlap_locomotoryinterneurons_permutations, axis = 2)
overlap_locomotoryinterneurons_permutations_std = np.std(overlap_locomotoryinterneurons_permutations, axis = 2)
overlap_vncmotor_permutations_mean = np.mean(overlap_vncmotor_permutations, axis = 2)
overlap_vncmotor_permutations_std = np.std(overlap_vncmotor_permutations, axis = 2)

overlap_pharynx_permutations_mean = np.mean(overlap_pharynx_permutations, axis = 2)
overlap_pharynx_permutations_std = np.std(overlap_pharynx_permutations, axis = 2)

overlap_ringinterneurons_permutations_mean = np.mean(overlap_ringinterneurons_permutations, axis = 2)
overlap_ringinterneurons_permutations_std = np.std(overlap_ringinterneurons_permutations, axis = 2)

overlap_allother_permutations_mean = np.mean(overlap_allother_permutations, axis = 2)
overlap_allother_permutations_std = np.std(overlap_allother_permutations, axis = 2)

overlap_rich_club_sp_permutations_mean = np.mean(overlap_rich_club_sp_permutations, axis = 2)
overlap_rich_club_sp_permutations_std = np.std(overlap_rich_club_sp_permutations, axis = 2)

###########calculating the z-scores and p-values

overlap_amphid_zscore = (overlap_amphid - overlap_amphid_permutations_mean)/overlap_amphid_permutations_std
overlap_amphid_pvalues = np.zeros((len(np.unique(groups_heirarchical)),1))

overlap_locomotoryinterneurons_zscore = (overlap_locomotoryinterneurons - overlap_locomotoryinterneurons_permutations_mean)/overlap_locomotoryinterneurons_permutations_std
overlap_locomotoryinterneurons_pvalues = np.zeros((len(np.unique(groups_heirarchical)),1))
overlap_vncmotor_zscore = (overlap_vncmotor - overlap_vncmotor_permutations_mean)/overlap_vncmotor_permutations_std
overlap_vncmotor_pvalues = np.zeros((len(np.unique(groups_heirarchical)),1))

overlap_pharynx_zscore = (overlap_pharynx - overlap_pharynx_permutations_mean)/overlap_pharynx_permutations_std
overlap_pharynx_pvalues = np.zeros((len(np.unique(groups_heirarchical)),1))

overlap_ringinterneurons_zscore = (overlap_ringinterneurons - overlap_ringinterneurons_permutations_mean)/overlap_ringinterneurons_permutations_std
overlap_ringinterneurons_pvalues = np.zeros((len(np.unique(groups_heirarchical)),1))

overlap_allother_zscore = (overlap_allother - overlap_allother_permutations_mean)/overlap_allother_permutations_std
overlap_allother_pvalues = np.zeros((len(np.unique(groups_heirarchical)),1))

overlap_rich_club_sp_zscore = (overlap_rich_club_sp - overlap_rich_club_sp_permutations_mean)/overlap_rich_club_sp_permutations_std
overlap_rich_club_sp_pvalues = np.zeros((len(np.unique(groups_heirarchical)),1))

for b in np.arange(len(groups_heirarchical_unique)):
    overlap_amphid_pvalues[b] = stats.ttest_ind(overlap_amphid[b], overlap_amphid_permutations[b, 0, :],
                                                  alternative="greater").pvalue
    overlap_locomotoryinterneurons_pvalues[b] = stats.ttest_ind(overlap_locomotoryinterneurons[b], overlap_locomotoryinterneurons_permutations[b, 0, :],
                                                alternative="greater").pvalue
    overlap_vncmotor_pvalues[b] = stats.ttest_ind(overlap_vncmotor[b], overlap_vncmotor_permutations[b, 0, :],
                                                    alternative="greater").pvalue
    overlap_pharynx_pvalues[b] = stats.ttest_ind(overlap_pharynx[b], overlap_pharynx_permutations[b, 0, :],
                                                 alternative="greater").pvalue
    overlap_ringinterneurons_pvalues[b] = stats.ttest_ind(overlap_ringinterneurons[b], overlap_ringinterneurons_permutations[b, 0, :],
                                                 alternative="greater").pvalue
    overlap_allother_pvalues[b] = stats.ttest_ind(overlap_allother[b], overlap_allother_permutations[b, 0, :],
                                                 alternative="greater").pvalue
    overlap_rich_club_sp_pvalues[b] = stats.ttest_ind(overlap_rich_club_sp[b], overlap_rich_club_sp_permutations[b, 0, :],
                                                  alternative="greater").pvalue


overlap_amphid_pvalues_flattened = overlap_amphid_pvalues.flatten()
overlap_amphid_pvalues_corrected = stats.false_discovery_control(overlap_amphid_pvalues_flattened)
overlap_amphid_pvalues_corrected_matrix = overlap_amphid_pvalues_corrected.reshape(len(groups_heirarchical_unique),1)

overlap_locomotoryinterneurons_pvalues_flattened = overlap_locomotoryinterneurons_pvalues.flatten()
overlap_locomotoryinterneurons_pvalues_corrected = stats.false_discovery_control(overlap_locomotoryinterneurons_pvalues_flattened)
overlap_locomotoryinterneurons_pvalues_corrected_matrix = overlap_locomotoryinterneurons_pvalues_corrected.reshape(len(groups_heirarchical_unique),1)

overlap_vncmotor_pvalues_flattened = overlap_vncmotor_pvalues.flatten()
overlap_vncmotor_pvalues_corrected = stats.false_discovery_control(overlap_vncmotor_pvalues_flattened)
overlap_vncmotor_pvalues_corrected_matrix = overlap_vncmotor_pvalues_corrected.reshape(len(groups_heirarchical_unique),1)


overlap_pharynx_pvalues_flattened = overlap_pharynx_pvalues.flatten()
overlap_pharynx_pvalues_corrected = stats.false_discovery_control(overlap_pharynx_pvalues_flattened)
overlap_pharynx_pvalues_corrected_matrix = overlap_pharynx_pvalues_corrected.reshape(len(groups_heirarchical_unique),1)


overlap_ringinterneurons_pvalues_flattened = overlap_ringinterneurons_pvalues.flatten()
overlap_ringinterneurons_pvalues_corrected = stats.false_discovery_control(overlap_ringinterneurons_pvalues_flattened)
overlap_ringinterneurons_pvalues_corrected_matrix = overlap_ringinterneurons_pvalues_corrected.reshape(len(groups_heirarchical_unique),1)


overlap_allother_pvalues_flattened = overlap_allother_pvalues.flatten()
overlap_allother_pvalues_corrected = stats.false_discovery_control(overlap_allother_pvalues_flattened)
overlap_allother_pvalues_corrected_matrix = overlap_allother_pvalues_corrected.reshape(len(groups_heirarchical_unique),1)


overlap_rich_club_sp_pvalues_flattened = overlap_rich_club_sp_pvalues.flatten()
overlap_rich_club_sp_pvalues_corrected = stats.false_discovery_control(overlap_rich_club_sp_pvalues_flattened)
overlap_rich_club_sp_pvalues_corrected_matrix = overlap_rich_club_sp_pvalues_corrected.reshape(len(groups_heirarchical_unique),1)


#create a matrix with all the categories, i should have done it like this to begin with but oh well
# 0: pharynx, 1: amphids, 2: locomotory interneurons, 3: vnc motor neurons, 4:ring, 5: all other
overlap_all = np.zeros((len(groups_heirarchical_unique),6))
overlap_all_zscore = np.zeros((len(groups_heirarchical_unique),6))
overlap_all_pvalues_corrected_matrix = np.zeros((len(groups_heirarchical_unique),6))

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

#plot the overlap
fig, ax = plt.subplots()
ax.imshow(overlap_all, cmap=plt.cm.Blues, vmax = 6)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(6):
        c = overlap_all[i,j]
        if  c < 6:
            ax.text(j, i, str(c), va='center', ha='center')
        else:
            ax.text(j, i, str(c), color = "white", va='center', ha='center')
plt.xticks(np.arange(6), ["Pharyngeal neurons", "Amphid sensory neurons", "Locomotory interneurons",
                          "VNC motorneurons", "Ring interneurons", "All others"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/neurontype_enrichment_mcmc_others.pdf", dpi=300, bbox_inches="tight")
plt.show()

#total number in each category
fig, ax = plt.subplots()
ax.imshow(np.reshape(np.sum(overlap_all, axis = 0), (1,5)), cmap=plt.cm.Blues)
for i in np.arange(len(np.sum(overlap_all, axis = 0))):
        c = np.sum(overlap_all, axis = 0)[i]
        if c < 24:
            ax.text(i, 0, str(c), va='center', ha='center')
        else:
            ax.text(i, 0, str(c), color = "white", va='center', ha='center')
plt.xticks(np.arange(5), ["Pharyngeal neurons", "Amphid sensory neurons", "Locomotory interneurons",
                          "VNC motorneurons", "Ring interneurons"])
ax.set_ylabel("Number")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/neurontype_enrichment_nr_mcmc.pdf", dpi=300, bbox_inches="tight")
plt.show()

#plot z-score
fig, ax = plt.subplots()
plt.imshow(overlap_all_zscore, cmap="coolwarm", vmin = -4, vmax = 4)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(6):
        if overlap_all_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(j, i, "**", va='center', ha='center')
        elif overlap_all_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(j, i, "*", va='center', ha='center')
plt.xticks(np.arange(6), ["Pharyngeal neurons", "Amphid sensory neurons", "Locomotory interneurons",
                          "VNC motorneurons", "Ring interneurons", "All others"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/neurontype_enrichment_zscores_mcmc_other.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig2/neurontype_enrichment_zscores_mcmc_other.eps",
    dpi=300, bbox_inches="tight", format="eps")
plt.show()

#plot p-values
fig, ax = plt.subplots()
plt.imshow(overlap_all_pvalues_corrected_matrix, cmap="Purples_r", vmin = 0, vmax = 1)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(5):
        if overlap_all_pvalues_corrected_matrix[i, j] < 0.01:
            ax.text(j, i, "**", va='center', color="white", ha='center')
        elif overlap_all_pvalues_corrected_matrix[i, j] < 0.05:
            ax.text(j, i, "*", va='center', color="white", ha='center')
plt.xticks(np.arange(5), ["Pharyngeal neurons", "Amphid sensory neurons", "Locomotory interneurons",
                          "VNC motorneurons", "Ring interneurons"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
plt.colorbar().ax.set_ylabel('corrected p-values', rotation=270, labelpad=15)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/neurontype_enrichment_pvals_mcmc.pdf", dpi=300, bbox_inches="tight")
plt.show()


#plot rich club z-score
fig, ax = plt.subplots()
plt.imshow(overlap_rich_club_sp_zscore, cmap="coolwarm", vmin = -4, vmax = 4)
for i in np.arange(len(groups_heirarchical_unique)):
    if overlap_rich_club_sp_pvalues_corrected_matrix[i] < 0.01:
        ax.text(0, i, "**", va='center', ha='center')
    elif overlap_rich_club_sp_pvalues_corrected_matrix[i] < 0.05:
        ax.text(0, i, "*", va='center', ha='center')
plt.xticks(np.arange(1), ["Signaling Rich Club"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/richclub_module_enrichment_zscores_mcmc.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig5/richclub_module_enrichment_zscores_mcmc.eps",
    dpi=300, bbox_inches="tight", format="eps")
plt.show()


print("done")