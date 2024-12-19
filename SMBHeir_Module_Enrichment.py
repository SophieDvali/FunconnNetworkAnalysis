import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import matplotlib
from matplotlib_venn import venn2



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


with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/fconn_state_mcmc.pickle', 'rb') as st:
    state = pickle.load(st)
with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/aconn_state_mcmc.pickle', 'rb') as st_a:
    state_aconn = pickle.load(st_a)




levels = state.get_levels()


heirarchical_level_0 = levels[0]

groups_heirarchical = []
for i in np.arange(len(neuronlist)):
    groups_heirarchical.append(levels[0].get_blocks()[i])


print(groups_heirarchical)
print(np.unique(groups_heirarchical))


group_lengths = []
for gr in np.unique(groups_heirarchical):
    group_lengths.append(len(np.where(groups_heirarchical == gr)[0]))
    
group_sorter = np.argsort(group_lengths)

#the order of the groups is determined by this so that they are always the same, 1-5
groups_heirarchical_unique = np.unique(groups_heirarchical)[group_sorter]

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

group_exp = 1
#calculating the overlap between motor sensory and inter with the modules
#1 is sensory, 2 is inter, 3 is motor
overlap_SIM = np.zeros((len(groups_heirarchical_unique),3))
overlap_pharynx = np.zeros((len(groups_heirarchical_unique),1))
for i in np.arange(len(groups_heirarchical_unique)):
    indicies_of_group = np.where(groups_heirarchical == groups_heirarchical_unique[i])[0]
    neurons_in_group = np.array(neuronlist)[indicies_of_group]
    if i == group_exp:
        group_exp_neurons = neurons_in_group
    for neuron in neurons_in_group:
        if neuron in sensory:
            overlap_SIM[i,0] += 1
        elif neuron in inter:
            overlap_SIM[i, 1] += 1
        elif neuron in motor:
            overlap_SIM[i, 2] += 1

        if neuron in pharynx:
            overlap_pharynx[i] += 1

percentage_SIM =  np.zeros((len(groups_heirarchical_unique),3))
percentage_pharynx = np.zeros((len(groups_heirarchical_unique),1))

for i in np.arange(len(groups_heirarchical_unique)):
    percentage_SIM[i, :] = overlap_SIM[i, :] / np.sum(overlap_SIM, axis=1)[i]
    percentage_pharynx[i] = overlap_pharynx[i]/ np.sum(overlap_SIM, axis=1)[i]

for bl in np.arange(len(groups_heirarchical_unique)):
    fig = plt.figure(figsize=(10, 7))
    plt.pie(percentage_SIM[bl,:], labels=["Sensory neurons", "Interneurons", "Motorneurons"],
            colors = ("green", "purple", "orange"))
    plt.title("Makeup of community "+str(bl+1))
    plt.tight_layout()
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_piechart_mcmc_block"+str(bl+1)+".pdf",
                dpi=300, bbox_inches="tight")
    plt.show()


fig = plt.figure(figsize=(10, 7))
plt.pie([len(sensory),len(inter),len(motor)], labels=["Sensory neurons", "Interneurons", "Motorneurons"],
        colors = ("green", "purple", "orange"))
plt.title("Makeup of all neurons")
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_piechart_mcmc_allneurons.pdf",
            dpi=300, bbox_inches="tight")
plt.show()




fig, ax = plt.subplots()
ax.imshow(overlap_SIM, cmap=plt.cm.Blues)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(3):
        c = overlap_SIM[i,j]
        ax.text(j, i, str(c), va='center', ha='center')
ax.set_xticklabels(["", "Sensoryneuron", "Interneuron", "Motorneuron"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment_mcmc.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment.pdf", dpi=300, bbox_inches="tight")

plt.show()

fig, ax = plt.subplots()
ax.imshow(percentage_SIM, cmap=plt.cm.Blues)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(3):
        c = percentage_SIM[i,j]
        ax.text(j, i, str(round(100*c,1))+"%", va='center', ha='center')
ax.set_xticklabels(["", "Sensoryneuron", "Interneuron", "Motorneuron"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment_percent_mcmc.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment_percent.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.imshow(percentage_pharynx, cmap=plt.cm.Blues)
for i in np.arange(len(groups_heirarchical_unique)):
        c2 = percentage_pharynx[i]
        ax.text(0, i, str(round(100*c2[0], 1))+"%", va='center', ha='center')
ax.set_xticklabels(["", "Pharyngeal neurons"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/pharynx_enrichment_percent_mcmc.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/pharynx_enrichment_percent.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.imshow(overlap_pharynx, cmap=plt.cm.Blues)
for i in np.arange(len(groups_heirarchical_unique)):
        c2 = overlap_pharynx[i]
        ax.text(0, i, str(c2[0]), va='center', ha='center')
ax.set_xticklabels(["", "Pharyngeal neurons"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/pharynx_enrichment_mcmc.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/pharynx_enrichment.pdf", dpi=300, bbox_inches="tight")
plt.show()
#do random permutations
#np.random.permutation
#lets start with 1000
overlap_SIM_permutations = np.zeros((len(groups_heirarchical_unique),3,1000))
overlap_pharynx_permutations = np.zeros((len(groups_heirarchical_unique),1, 1000))
for r in np.arange(1000):
    groups_heirarchical_permute = np.random.permutation(groups_heirarchical)
    for i in np.arange(len(groups_heirarchical_unique)):
        indicies_of_group = np.where(groups_heirarchical_permute == groups_heirarchical_unique[i])[0]
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


overlap_SIM_permutations_mean = np.mean(overlap_SIM_permutations, axis = 2)
overlap_SIM_permutations_std = np.std(overlap_SIM_permutations, axis = 2)

overlap_pharynx_permutations_mean = np.mean(overlap_pharynx_permutations, axis = 2)
overlap_pharynx_permutations_std = np.std(overlap_pharynx_permutations, axis = 2)

overlap_SIM_zscore = (overlap_SIM - overlap_SIM_permutations_mean)/overlap_SIM_permutations_std
overlap_SIM_pvalues = np.zeros((len(np.unique(groups_heirarchical)),3))

overlap_pharynx_zscore = (overlap_pharynx - overlap_pharynx_permutations_mean)/overlap_pharynx_permutations_std
overlap_pharynx_pvalues = np.zeros((len(np.unique(groups_heirarchical)),1))

for b in np.arange(len(groups_heirarchical_unique)):
    for sim in np.arange(3):
        overlap_SIM_pvalues[b,sim] = stats.ttest_ind(overlap_SIM[b,sim], overlap_SIM_permutations[b,sim,:], alternative = "greater").pvalue
    overlap_pharynx_pvalues[b] = stats.ttest_ind(overlap_pharynx[b], overlap_pharynx_permutations[b, 0, :],
                                                  alternative="greater").pvalue
overlap_SIM_pvalues_flattened = overlap_SIM_pvalues.flatten()
overlap_SIM_pvalues_corrected = stats.false_discovery_control(overlap_SIM_pvalues_flattened)
overlap_SIM_pvalues_corrected_matrix = overlap_SIM_pvalues_corrected.reshape(len(groups_heirarchical_unique),3)

overlap_pharynx_pvalues_flattened = overlap_pharynx_pvalues.flatten()
overlap_pharynx_pvalues_corrected = stats.false_discovery_control(overlap_pharynx_pvalues_flattened)
overlap_pharynx_pvalues_corrected_matrix = overlap_pharynx_pvalues_corrected.reshape(len(groups_heirarchical_unique),1)

fig, ax = plt.subplots()
plt.imshow(overlap_SIM_zscore, cmap="coolwarm", vmin = -4, vmax = 4)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(3):
        if overlap_SIM_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(j, i, "**", va='center', ha='center')
        elif overlap_SIM_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(j, i, "*", va='center', ha='center')
ax.set_xticklabels(["", "Sensoryneuron", "Interneuron", "Motorneuron"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment_zscores_mcmc.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig2/SIM_enrichment_zscores_mcmc.eps",
        dpi=300, bbox_inches="tight", format ="eps")

else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment_zscores.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
plt.imshow(overlap_SIM_pvalues_corrected_matrix,  cmap="Purples_r", vmax = 1, vmin = 0)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(3):
        if overlap_SIM_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(j, i, "**", color="white", va='center', ha='center')
        elif overlap_SIM_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(j, i, "*", color="white", va='center', ha='center')
ax.set_xticklabels(["", "Sensoryneuron", "Interneuron", "Motorneuron"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
#cbar = plt.colorbar()
#cbar.set_label('corrected p-values', rotation=270)
plt.colorbar().ax.set_ylabel('corrected p-values', rotation=270, labelpad=15)
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment_pvals_mcmc.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment_pvals.pdf", dpi=300, bbox_inches="tight")
plt.show()


fig, ax = plt.subplots()
plt.imshow(overlap_pharynx_zscore, cmap="coolwarm", vmin = -4, vmax = 4)
for i in np.arange(len(groups_heirarchical_unique)):
    if overlap_pharynx_pvalues_corrected_matrix[i] < 0.01:
        ax.text(0, i, "**", va='center', ha='center')
    elif overlap_pharynx_pvalues_corrected_matrix[i] < 0.05:
        ax.text(0, i, "*", va='center', ha='center')
ax.set_xticklabels(["", "Pharyngeal neurons"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/pharynx_enrichment_zscores_mcmc.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/pharynx_enrichment_zscores.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
plt.imshow(overlap_pharynx_pvalues_corrected_matrix, cmap="Purples_r")
for i in np.arange(len(groups_heirarchical_unique)):
    if overlap_pharynx_pvalues_corrected_matrix[i] < 0.01:
        ax.text(0, i, "**", color="white", va='center', ha='center')
    elif overlap_pharynx_pvalues_corrected_matrix[i] < 0.05:
        ax.text(0, i, "*", color="white", va='center', ha='center')
ax.set_xticklabels(["", "Pharyngeal neurons"])
plt.yticks(np.arange(6), np.arange(6)+1)
ax.set_ylabel("Block")
plt.xticks(rotation=90)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('corrected p-values', rotation=270)
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/pharynx_enrichment_pvals_mcmc.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/pharynx_enrichment_pvals.pdf", dpi=300, bbox_inches="tight")
plt.show()

##### figures for explaining the method we will use block 2 (1) and motor neurons

set1 = set(group_exp_neurons)
set2 = set(sensory)
venn2([set1, set2], ('Community 2', 'Sensory Neurons'), set_colors=("grey",
                             "green"))
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichemnt_venn_digram.pdf", dpi=300, bbox_inches="tight")

plt.show()

group_exp_permutations = overlap_SIM_permutations[group_exp,:,:]

df_group_exp_permutations = pd.DataFrame(group_exp_permutations.transpose())
df_group_exp_permutations.head()

vals, names, xs = [],[],[]
for i, col in enumerate(df_group_exp_permutations.columns):
    vals.append(df_group_exp_permutations[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, df_group_exp_permutations[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

plt.figure()
plt.boxplot(vals, labels=["Sensory", "Inter", "Motor"])
palette = ['darkgrey', 'darkgrey', 'darkgrey']
for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)
plt.plot([1],overlap_SIM[group_exp,0], marker = '_', markersize = 45, linestyle = "", markeredgewidth = 4, color = "orangered")
plt.plot([2,3],overlap_SIM[group_exp,1:], marker = '_', markersize = 45, linestyle = "", markeredgewidth = 4, color = "cornflowerblue")
#plt.plot([1,2,3],overlap_SIM_permutations_mean[group_exp,:], marker = '_', markersize = 45, linestyle = "", markeredgewidth = 4, color = "dimgrey")
plt.ylabel('Overlap with Community '+ str(group_exp+1), fontsize = 15)
plt.xticks([1,2,3], ["Sensory", "Inter", "Motor"], fontsize = 15)
plt.yticks([0,5,10,15], [0,5,10,15],fontsize = 15 )
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichemnt_exp_scatter.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig2/SIM_enrichemnt_exp_scatter.eps",
        dpi=300, bbox_inches="tight", format ="eps")
plt.show()

plt.figure()
violin_parts = plt.violinplot(group_exp_permutations.transpose(), showmeans = True)
for pc in violin_parts['bodies']:
    pc.set_facecolor('darkgrey')
    pc.set_edgecolor('darkgrey')
# Make all the violin statistics marks red:
for partname in ('cbars','cmins','cmaxes', "cmeans"):
    vp = violin_parts[partname]
    vp.set_edgecolor("darkgrey")
    vp.set_linewidth(1)
plt.plot([1],overlap_SIM[group_exp,0], marker = '_', markersize = 45, linestyle = "", markeredgewidth = 4, color = "orangered")
plt.plot([2,3],overlap_SIM[group_exp,1:], marker = '_', markersize = 45, linestyle = "", markeredgewidth = 4, color = "cornflowerblue")
plt.ylabel('Overlap with Community '+ str(group_exp+1), fontsize = 15)
plt.xticks([1,2,3], ["Sensory", "Inter", "Motor"], fontsize = 15)
plt.yticks([0,5,10,15], [0,5,10,15],fontsize = 15 )
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichemnt_exp_scatter.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig2/SIM_enrichemnt_exp_scatter.eps",
        dpi=300, bbox_inches="tight", format ="eps")
plt.show()

number_connections = np.zeros((len(groups_heirarchical_unique),len(groups_heirarchical_unique)))
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(len(groups_heirarchical_unique)):
        indicies_in_i= np.where(groups_heirarchical == groups_heirarchical_unique[i])[0]
        neurons_in_i = np.array(neuronlist)[indicies_in_i]
        indicies_in_j= np.where(groups_heirarchical == groups_heirarchical_unique[j])[0]
        neurons_in_j = np.array(neuronlist)[indicies_in_j]

        for neu_i in indicies_in_i:
            for neu_j in indicies_in_j:
                if g.edge(neu_j,neu_i):
                    number_connections[i,j] += 1

normalized_by_self_connections = np.copy(number_connections)
selfconnections = np.diagonal(number_connections)


total_outgoing = np.sum(number_connections, axis = 0)
total_ingoing = np.sum(number_connections, axis = 1)

percent_outgoing =  number_connections/total_outgoing[None, :]
percent_ingoing =  number_connections/total_ingoing[:, None]


normalized_by_nr_neurons = np.zeros((len(groups_heirarchical_unique),len(groups_heirarchical_unique)))
group_lengths_sorted = np.array(group_lengths)[group_sorter]
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(len(groups_heirarchical_unique)):
        number_neu_for_groups = group_lengths_sorted[i] * group_lengths_sorted[j] #use multiplication instead
        normalized_by_nr_neurons[i,j] = number_connections[i,j]/number_neu_for_groups



binary_nr_connections = np.copy(number_connections)
binary_nr_connections[binary_nr_connections != 0] = 1

nr_connections_adj_matrix = {}
for j in np.arange(len(groups_heirarchical_unique)):
    nr_connections_adj_matrix[j] = np.where(binary_nr_connections[:,j] != 0)[0]

bg = gt.Graph(nr_connections_adj_matrix)
eprop_bg = bg.new_edge_property("double")                # Arbitrary Python object.
bg.ep.weight = eprop_bg

eprop_bg_2 = bg.new_edge_property("double")                # Arbitrary Python object.
bg.ep.weight_2 = eprop_bg_2

vprop2_bg = bg.new_vertex_property("int")
bg.vp.number = vprop2_bg

vcolor = bg.new_vertex_property("string")
bg.vp.color = vcolor

n = 0
for v in bg.vertices():
    bg.vp.number[v] = n
    n+=1
    if v in [1,5]:
        bg.vp.color[v] = "#38761D"
    elif v == 3:
        bg.vp.color[v] = "#0B5394"
    elif v in [0,2]:
        bg.vp.color[v] = "#CC0000"
    else:
        bg.vp.color[v] = "#5B5B5B"




for v in bg.vertices():
    for e in v.out_edges():
        #eprop[e] = mappa_connected_only[g.vp.number[e.target()], g.vp.number[e.source()]]
        bg.ep.weight[e] = (number_connections[bg.vp.number[e.target()], bg.vp.number[e.source()]]/np.max(number_connections))*10
        bg.ep.weight_2[e] = (normalized_by_nr_neurons[bg.vp.number[e.target()], bg.vp.number[e.source()]])*4


gt.graph_draw(bg, vertex_text=bg.vertex_index, edge_pen_width=bg.ep.weight, vertex_fill_color=bg.vp.color,  output="blocksgraph.pdf")




neurons_in_5 = np.where(groups_heirarchical == groups_heirarchical_unique[5])[0]
total_connections_5_to_5 = 0
neg_connections_in_5 = 0
for i in neurons_in_5:
    for j in neurons_in_5:
        if mappa_connected_only[i,j]:
            total_connections_5_to_5 += 1
            if mappa_connected_only[i,j] < 0:
                neg_connections_in_5 += 1


total_connections_b_to_b = np.zeros((len(np.unique(groups_heirarchical)),1))
neg_connections_in_b = np.zeros((len(np.unique(groups_heirarchical)),1))

for b in np.arange(len(np.unique(groups_heirarchical))):
    neurons_in_b = np.where(groups_heirarchical == groups_heirarchical_unique[b])[0]
    for i in neurons_in_b:
        for j in neurons_in_b:
            if mappa_connected_only[i, j]:
                total_connections_b_to_b[b] += 1
                if mappa_connected_only[i, j] < 0:
                    neg_connections_in_b[b] += 1



fig, ax = plt.subplots()
plt.imshow(number_connections, cmap="viridis", vmax = 100)
#ax.set_xticklabels(["", "Pharyngeal neurons"])
ax.set_xlabel("From Block")
ax.set_ylabel("To Block")
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('Number of Edges', rotation=270)
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/block_to_block_edges_mcmc.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/block_to_block_edges.pdf", dpi=300, bbox_inches="tight")
plt.show()

#blocks 1,2,5 are more sensory, block 4 is more inter, block 0, 3 are more motor for the exclude unconnected
#for regular mcmc 1,5 are sensory, 3 is inter, 0 and 2 are motor and 4 is uncategorized
if exclude_unconnected:
    SIM_block_sorter = [1,2,5,4,0,3]
else:
    SIM_block_sorter = [1, 5, 3, 0, 2, 4]

SIM_sorted_number_connections = number_connections[:, SIM_block_sorter][SIM_block_sorter]
SIM_sorted_percent_outgoing =  percent_outgoing[:, SIM_block_sorter][SIM_block_sorter]
SIM_sorted_percent_ingoing =  percent_ingoing[:, SIM_block_sorter][SIM_block_sorter]

ticks_label = SIM_block_sorter.copy()
ticks_label.insert(0, " ")


fig, ax = plt.subplots()
plt.imshow(SIM_sorted_percent_outgoing, cmap="viridis")
ax.set_xticklabels(ticks_label)
ax.set_yticklabels(ticks_label)
ax.set_xlabel("To Block")
ax.set_ylabel("From Block")
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('Fraction of outgoing edges', rotation=270)
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/block_to_block_edges_frac_outgoing_sorted_mcmc.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/block_to_block_sorted_frac_outgoing_edges.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
plt.imshow(SIM_sorted_percent_ingoing, cmap="viridis")
ax.set_xticklabels(ticks_label)
ax.set_yticklabels(ticks_label)
ax.set_xlabel("To Block")
ax.set_ylabel("From Block")
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('Fraction of incoming edges', rotation=270)
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/block_to_block_edges_frac_ingoing_sorted_mcmc.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/block_to_block_sorted_frac_ingoing_edges.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
plt.imshow(SIM_sorted_number_connections, cmap="viridis", vmax = 100)
ax.set_xticklabels(ticks_label)
ax.set_yticklabels(ticks_label)
ax.set_xlabel("To Block")
ax.set_ylabel("From Block")
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('Number of Edges', rotation=270)
if state_mcmc:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/block_to_block_edges_sorted_mcmc.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/block_to_block_sorted_edges.pdf", dpi=300, bbox_inches="tight")
plt.show()

print("done")
