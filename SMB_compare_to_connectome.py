import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from matplotlib_venn import venn2
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
#import umap
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score
#def SIGSEGV_signal_arises(signalNum, stack): pass
#signal.signal(signal.SIGSEGV, SIGSEGV_signal_arises)
from sklearn.metrics import mutual_info_score
from scipy import stats
import sys
import matplotlib
from munkres import Munkres
from scipy.stats import linregress

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
eprop = g.new_edge_property("double")                # Arbitrary Python object.
g.ep.weight = eprop
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


def get_sbm_stats(state,n):

    levels = state.get_levels()

    nlevels = len(levels)

    lbls = np.zeros([n,nlevels])

    for i in range(n):

        r = levels[0].get_blocks()[i]

        lbls[i,0] = r

        for j in range(1,nlevels):

            r = levels[j].get_blocks()[r]

            lbls[i,j] = r

    mdl = state.entropy()

    return lbls,mdl



levels = state.get_levels()
for s in levels:
    print(s)
    if s.get_N() == 1:
        break

levels_aconn = state_aconn.get_levels()

groups_heirarchical_all_aconn = get_sbm_stats(state_aconn, 188)

groups_heirarchical = []
for i in np.arange(len(neuronlist)):
    groups_heirarchical.append(levels[0].get_blocks()[i])

groups_heirarchical_aconn = []
for i in np.arange(188):
    groups_heirarchical_aconn.append(levels_aconn[0].get_blocks()[i])

#groups_heirarchical_aconn_level1 = groups_heirarchical_all_aconn[0][:,1]

groups_heirarchical_dict = {}
for i in np.unique(groups_heirarchical):
    groups_heirarchical_dict[i] = np.where(np.array(groups_heirarchical) == i)[0]

group_lengths = []
for gr in np.unique(groups_heirarchical):
    group_lengths.append(len(np.where(groups_heirarchical == gr)[0]))

group_sorter = np.argsort(group_lengths)

# the order of the groups is determined by this so that they are always the same, by size block 0 is smallest
groups_heirarchical_unique = np.unique(groups_heirarchical)[group_sorter]

group_lengths_aconn = []
for gr in np.unique(groups_heirarchical_aconn):
    group_lengths_aconn.append(len(np.where(groups_heirarchical_aconn == gr)[0]))
group_sorter_aconn = np.argsort(group_lengths_aconn)

groups_heirarchical_unique_aconn = np.unique(groups_heirarchical_aconn)[group_sorter_aconn]

# group_lengths_aconn_level1 = []
# for gr in np.unique(groups_heirarchical_aconn_level1):
#     group_lengths_aconn_level1.append(len(np.where(groups_heirarchical_aconn_level1 == gr)[0]))
# group_sorter_aconn_level1 = np.argsort(group_lengths_aconn_level1)
#
# groups_heirarchical_unique_aconn_level1 = np.unique(groups_heirarchical_aconn_level1)[group_sorter_aconn_level1]


#calculating the overlap between anatomical and functional
overlap_aconn_level0 = np.zeros((len(groups_heirarchical_unique),len(groups_heirarchical_unique_aconn)))
overlap_aconn_level1 = np.zeros((len(groups_heirarchical_unique),len(groups_heirarchical_unique_aconn_level1)))
for neu in np.arange(len(groups_heirarchical)):
    group_funconn = np.where(groups_heirarchical_unique == groups_heirarchical[neu])[0]
    group_aconn_level0 = np.where(groups_heirarchical_unique_aconn == groups_heirarchical_aconn[neu])[0]
    group_aconn_level1 = np.where(groups_heirarchical_unique_aconn_level1 == groups_heirarchical_aconn_level1[neu])[0]
    overlap_aconn_level0[group_funconn,group_aconn_level0] +=1
    overlap_aconn_level1[group_funconn, group_aconn_level1] += 1



fig, ax = plt.subplots()
ax.imshow(overlap_aconn_level0, cmap=plt.cm.Blues)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(len(groups_heirarchical_unique_aconn)):
        c = overlap_aconn_level0[i,j]
        ax.text(j, i, str(int(c)), va='center', ha='center')
ax.set_ylabel("Block Fconn")
ax.set_xlabel("Block Aconn")
plt.xticks(ticks = np.arange(len(groups_heirarchical_unique_aconn)), labels =  np.arange(len(groups_heirarchical_unique_aconn)))
plt.yticks(ticks = np.arange(len(groups_heirarchical_unique)), labels = np.arange(len(groups_heirarchical_unique)))
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_enrichment_mcmc.pdf", dpi=300, bbox_inches="tight")
plt.show()
#
# fig, ax = plt.subplots()
# ax.imshow(overlap_aconn_level1, cmap=plt.cm.Blues)
# for i in np.arange(len(groups_heirarchical_unique)):
#     for j in np.arange(len(groups_heirarchical_unique_aconn_level1)):
#         c = overlap_aconn_level1[i,j]
#         ax.text(j, i, str(int(c)), va='center', ha='center')
# ax.set_ylabel("Block Fconn")
# ax.set_xlabel("Block Aconn")
# plt.tight_layout()
# plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level1_enrichment_mcmc.pdf", dpi=300, bbox_inches="tight")
# plt.show()

#lets start with 500
overlap_aconn_level0_permutations = np.zeros((len(groups_heirarchical_unique),len(groups_heirarchical_unique_aconn),1000))
#overlap_aconn_level1_permutations = np.zeros((len(groups_heirarchical_unique),len(groups_heirarchical_unique_aconn_level1), 1000))
for r in np.arange(1000):
    groups_heirarchical_permute = np.random.permutation(groups_heirarchical)
    for neu in np.arange(len(groups_heirarchical)):
        group_funconn = np.where(groups_heirarchical_unique == groups_heirarchical_permute[neu])[0]
        group_aconn_level0 = np.where(groups_heirarchical_unique_aconn == groups_heirarchical_aconn[neu])[0]
        #group_aconn_level1 = np.where(groups_heirarchical_unique_aconn_level1 == groups_heirarchical_aconn_level1[neu])[
        #    0]
        overlap_aconn_level0_permutations[group_funconn, group_aconn_level0, r] += 1
        overlap_aconn_level1_permutations[group_funconn, group_aconn_level1, r] += 1

overlap_aconn_level0_permutations_mean = np.mean(overlap_aconn_level0_permutations, axis = 2)
overlap_aconn_level0_permutations_std = np.std(overlap_aconn_level0_permutations, axis = 2)

# overlap_aconn_level1_permutations_mean = np.mean(overlap_aconn_level1_permutations, axis = 2)
# overlap_aconn_level1_permutations_std = np.std(overlap_aconn_level1_permutations, axis = 2)

overlap_aconn_level0_zscore = (overlap_aconn_level0 - overlap_aconn_level0_permutations_mean)/overlap_aconn_level0_permutations_std
overlap_aconn_level0_pvalues = np.zeros((len(groups_heirarchical_unique),len(groups_heirarchical_unique_aconn)))

# overlap_aconn_level1_zscore = (overlap_aconn_level1 - overlap_aconn_level1_permutations_mean)/overlap_aconn_level1_permutations_std
# overlap_aconn_level1_pvalues = np.zeros((len(groups_heirarchical_unique),len(groups_heirarchical_unique_aconn_level1)))

for f in np.arange(len(groups_heirarchical_unique)):
    for a in np.arange(len(groups_heirarchical_unique_aconn)):
        if np.isnan(overlap_aconn_level0_zscore[f,a]):
            overlap_aconn_level0_pvalues[f, a] = np.nan
        else:
            overlap_aconn_level0_pvalues[f,a] = stats.ttest_ind(overlap_aconn_level0[f,a], overlap_aconn_level0_permutations[f,a,:], alternative = "greater").pvalue
    # for a1 in np.arange(len(groups_heirarchical_unique_aconn_level1)):
    #     overlap_aconn_level1_pvalues[f,a1] = stats.ttest_ind(overlap_aconn_level1[f,a1], overlap_aconn_level1_permutations[f,a1,:], alternative = "greater").pvalue

overlap_aconn_level0_pvalues_flattened = overlap_aconn_level0_pvalues.flatten()
overlap_aconn_level0_pvalues_corrected = stats.false_discovery_control(overlap_aconn_level0_pvalues_flattened[~np.isnan(overlap_aconn_level0_pvalues_flattened)])
overlap_aconn_level0_pvalues_corrected_new = np.zeros((1,len(overlap_aconn_level0_pvalues_flattened)))
overlap_aconn_level0_pvalues_corrected_new = overlap_aconn_level0_pvalues_corrected_new*np.NaN
overlap_aconn_level0_pvalues_corrected_new[0][~np.isnan(overlap_aconn_level0_pvalues_flattened)]  = overlap_aconn_level0_pvalues_corrected
overlap_aconn_level0_pvalues_corrected_matrix = overlap_aconn_level0_pvalues_corrected_new.reshape(len(groups_heirarchical_unique),len(groups_heirarchical_unique_aconn))

# overlap_aconn_level1_pvalues_flattened = overlap_aconn_level1_pvalues.flatten()
# overlap_aconn_level1_pvalues_corrected = stats.false_discovery_control(overlap_aconn_level1_pvalues_flattened)
# overlap_aconn_level1_pvalues_corrected_matrix = overlap_aconn_level1_pvalues_corrected.reshape(len(groups_heirarchical_unique),len(groups_heirarchical_unique_aconn_level1))


##### figures for explaining the method we will use block I and 20
group_exp = 0
group_exp_aconn = 19
set1 = set(neuronlist[np.where(groups_heirarchical == groups_heirarchical_unique[group_exp])[0]])
set2 = set(neuronlist[np.where(groups_heirarchical_aconn == groups_heirarchical_unique_aconn[group_exp_aconn])[0]])
venn2([set1, set2], ('Community I', 'Community 20'), set_colors=("#98CE87",
                             "#1DA149"))
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/aconn_enrichemnt_venn_digram.pdf", dpi=300, bbox_inches="tight")

plt.show()

group_exp_permutations = overlap_aconn_level0_permutations[group_exp,:,:]

df_group_exp_permutations = pd.DataFrame(group_exp_permutations.transpose())
df_group_exp_permutations.head()

vals, names, xs = [],[],[]
for i, col in enumerate(df_group_exp_permutations.columns):
    vals.append(df_group_exp_permutations[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, df_group_exp_permutations[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


plt.figure(figsize=(15, 7))
violin_parts = plt.violinplot(group_exp_permutations.transpose(), showmeans = True)
for pc in violin_parts['bodies']:
    pc.set_facecolor('darkgrey')
    pc.set_edgecolor('darkgrey')
# Make all the violin statistics marks red:
for partname in ('cbars','cmins','cmaxes', "cmeans"):
    vp = violin_parts[partname]
    vp.set_edgecolor("darkgrey")
    vp.set_linewidth(1)
plt.plot([20],overlap_aconn_level0[group_exp,19], marker = '_', markersize = 20, linestyle = "", markeredgewidth = 4, color = "orangered")
plt.plot(np.arange(19)+1,overlap_aconn_level0[group_exp,0:19], marker = '_', markersize = 20, linestyle = "", markeredgewidth = 4, color = "cornflowerblue")
plt.ylabel('Overlap with Community '+ str(group_exp+1), fontsize = 30)
plt.xlabel('Anatomical Community ', fontsize = 30)
plt.xticks(np.arange(20)+1, np.arange(20)+1, fontsize = 25)
plt.yticks([0,5,10,15], [0,5,10,15],fontsize = 25 )
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/aconn_enrichemnt_exp_scatter.pdf", dpi=300, bbox_inches="tight")
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/aconn_enrichemnt_exp_scatter.pdf", dpi=300, bbox_inches="tight")

plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/aconn_enrichemnt_exp_scatter.eps",
        dpi=300, bbox_inches="tight", format ="eps")
plt.show()

#######figures


fig, ax = plt.subplots()
plt.imshow(overlap_aconn_level0_zscore, cmap="coolwarm", vmin = -4, vmax = 4)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(len(groups_heirarchical_unique_aconn)):
        if overlap_aconn_level0_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(j, i, "**", va='center', ha='center')
        elif overlap_aconn_level0_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(j, i, "*", va='center', ha='center')
ax.set_ylabel("Block Fconn")
ax.set_xlabel("Block Aconn")
plt.xticks(ticks = np.arange(len(groups_heirarchical_unique_aconn)), labels =  np.arange(len(groups_heirarchical_unique_aconn)))
plt.yticks(ticks = np.arange(len(groups_heirarchical_unique)), labels = np.arange(len(groups_heirarchical_unique)))
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_enrich_zscore_mcmc.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
plt.imshow(overlap_aconn_level0_pvalues_corrected_matrix, cmap="Purples_r", vmin = 0, vmax = 1)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(len(groups_heirarchical_unique_aconn)):
        if overlap_aconn_level0_pvalues_corrected_matrix[i,j] < 0.01:
            ax.text(j, i, "**", color="white", va='center', ha='center')
        elif overlap_aconn_level0_pvalues_corrected_matrix[i,j] < 0.05:
            ax.text(j, i, "*", color="white", va='center', ha='center')
ax.set_ylabel("Block Fconn")
ax.set_xlabel("Block Aconn")
plt.xticks(ticks = np.arange(len(groups_heirarchical_unique_aconn)), labels =  np.arange(len(groups_heirarchical_unique_aconn)))
plt.yticks(ticks = np.arange(len(groups_heirarchical_unique)), labels = np.arange(len(groups_heirarchical_unique)))
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('corrected p-values', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_enrich_pvals_mcmc.pdf", dpi=300, bbox_inches="tight")
plt.show()


Fs = np.zeros((len(groups_heirarchical_unique),len(groups_heirarchical)))

for gr in np.arange(len(groups_heirarchical_unique)):
    Fs[gr, np.where(np.array(groups_heirarchical) == groups_heirarchical_unique[gr])[0]] = 1

As = np.zeros((len(groups_heirarchical_unique_aconn),len(groups_heirarchical_aconn)))

for gr in np.arange(len(groups_heirarchical_unique_aconn)):
    As[gr, np.where(np.array(groups_heirarchical_aconn) == groups_heirarchical_unique_aconn[gr])[0]] = 1


similarity_aconn_jaccard = np.zeros((len(groups_heirarchical_unique),len(groups_heirarchical_unique_aconn)))
similarity_aconn_mutualinfo = np.zeros((len(groups_heirarchical_unique),len(groups_heirarchical_unique_aconn)))

for f in np.arange(len(groups_heirarchical_unique)):
    for a in np.arange(len(groups_heirarchical_unique_aconn)):
        similarity_aconn_jaccard[f,a] = jaccard_score(Fs[f,:], As[a,:])
        similarity_aconn_mutualinfo[f,a] = mutual_info_score(Fs[f,:], As[a,:])


fig, ax = plt.subplots()
plt.imshow(similarity_aconn_jaccard, cmap=plt.cm.Blues)
ax.set_ylabel("Block Fconn")
ax.set_xlabel("Block Aconn")
cbar = plt.colorbar()
cbar.set_label('jaccard score', rotation=270)
plt.xticks(ticks = np.arange(len(groups_heirarchical_unique_aconn)), labels =  np.arange(len(groups_heirarchical_unique_aconn)))
plt.yticks(ticks = np.arange(len(groups_heirarchical_unique)), labels = np.arange(len(groups_heirarchical_unique)))
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Similarity_Jaccard_aconn_mcmc.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
plt.imshow(similarity_aconn_mutualinfo, cmap=plt.cm.Blues)
ax.set_ylabel("Block Fconn")
ax.set_xlabel("Block Aconn")
cbar = plt.colorbar()
cbar.set_label('Mutual Information', rotation=270)
plt.xticks(ticks = np.arange(len(groups_heirarchical_unique_aconn)), labels =  np.arange(len(groups_heirarchical_unique_aconn)))
plt.yticks(ticks = np.arange(len(groups_heirarchical_unique)), labels = np.arange(len(groups_heirarchical_unique)))
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Similarity_MutualInfo_aconn_mcmc.pdf", dpi=300, bbox_inches="tight")
plt.show()

#calculate the corresponding rows and columns using the hungarian algorithm, 1- is to maximize the profit instead of the cost
mu = Munkres()
best_idxs_MI = mu.compute(1-similarity_aconn_mutualinfo)
best_idxs_J = mu.compute(1-similarity_aconn_jaccard)


sorterJ = []
sorterMI = []
for i in np.arange(len(best_idxs_J)):
    sorterJ.append(best_idxs_J[i][1])
    sorterMI.append(best_idxs_MI[i][1])

additional_indicies_J = np.where(~np.isin(np.arange(len(groups_heirarchical_unique_aconn)), sorterJ))[0]
additional_indicies_MI = np.where(~np.isin(np.arange(len(groups_heirarchical_unique_aconn)), sorterMI))[0]

sorterJ = np.concatenate((np.array(sorterJ),additional_indicies_J),axis = 0)
sorterMI = np.concatenate((np.array(sorterMI),additional_indicies_MI),axis = 0)

rearanged_similarity_aconn_jaccard = similarity_aconn_jaccard[:,sorterJ]
rearanged_similarity_aconn_mutualinfo = similarity_aconn_mutualinfo[:,sorterMI]
rearanged_pvals_aconn_jaccard = overlap_aconn_level0_pvalues_corrected_matrix[:,sorterJ]
rearanged_zscores_aconn_jaccard = overlap_aconn_level0_zscore[:,sorterJ]
rearanged_pvals_aconn_MI = overlap_aconn_level0_pvalues_corrected_matrix[:,sorterMI]
rearanged_zscores_aconn_MI = overlap_aconn_level0_zscore[:,sorterMI]


fig, ax = plt.subplots()
plt.imshow(rearanged_similarity_aconn_jaccard, cmap=plt.cm.Blues)
ax.set_ylabel("Block Fconn")
ax.set_xlabel("Block Aconn")
cbar = plt.colorbar()
cbar.set_label('jaccard score', rotation=270)
plt.xticks(ticks = np.arange(len(groups_heirarchical_unique_aconn)), labels = sorterJ)
plt.yticks(ticks = np.arange(len(groups_heirarchical_unique)), labels = np.arange(len(groups_heirarchical_unique)))
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Similarity_Jaccard_aconn_rearranged_mcmc.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
plt.imshow(rearanged_similarity_aconn_mutualinfo, cmap=plt.cm.Blues)
ax.set_ylabel("Block Fconn")
ax.set_xlabel("Block Aconn")
cbar = plt.colorbar()
cbar.set_label('mutual info', rotation=270)
plt.xticks(ticks = np.arange(len(groups_heirarchical_unique_aconn)), labels = sorterMI)
plt.yticks(ticks = np.arange(len(groups_heirarchical_unique)), labels = np.arange(len(groups_heirarchical_unique)))
plt.tight_layout()
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Similarity_mutualinfo_aconn_rearranged_mcmc.pdf",
        dpi=300, bbox_inches="tight")
plt.show()



fig, ax = plt.subplots()
plt.imshow(rearanged_pvals_aconn_jaccard, cmap="Purples_r", vmin = 0, vmax= 1)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(len(groups_heirarchical_unique_aconn)):
        if rearanged_pvals_aconn_jaccard[i,j] < 0.01:
            ax.text(j, i, "**", color="white", va='center', ha='center')
        elif rearanged_pvals_aconn_jaccard[i,j] < 0.05:
            ax.text(j, i, "*", color="white", va='center', ha='center')
ax.set_ylabel("Block Fconn")
ax.set_xlabel("Block Aconn")
plt.xticks(ticks = np.arange(len(groups_heirarchical_unique_aconn)), labels = sorterJ+1)
plt.yticks(ticks = np.arange(len(groups_heirarchical_unique)), labels = np.arange(len(groups_heirarchical_unique))+1)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('corrected p-values', rotation=270)
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_enrich_pvals_jaccardsorted_mcmc.pdf",
        dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
plt.imshow(rearanged_zscores_aconn_jaccard, cmap="coolwarm", vmin = -4, vmax = 4)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(len(groups_heirarchical_unique_aconn)):
        if rearanged_pvals_aconn_jaccard[i,j] < 0.01:
            ax.text(j, i, "**", va='center', ha='center')
        elif rearanged_pvals_aconn_jaccard[i,j] < 0.05:
            ax.text(j, i, "*", va='center', ha='center')
ax.set_ylabel("Block Fconn")
ax.set_xlabel("Block Aconn")
plt.xticks(ticks = np.arange(len(groups_heirarchical_unique_aconn)), labels = sorterJ+1)
plt.yticks(ticks = np.arange(len(groups_heirarchical_unique)), labels = np.arange(len(groups_heirarchical_unique))+1)
plt.tight_layout()
cbar = plt.colorbar()
cbar.set_label('z-score', rotation=270)
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_enrich_zscore_jaccardsorted_mcmc.pdf",
    dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/Aconn_level0_enrich_zscore_jaccardsorted_mcmc.eps",
    dpi=300, bbox_inches="tight", format="eps")
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_enrich_zscore_jaccardsorted.pdf", dpi=300, bbox_inches="tight")
plt.show()

rearanged_overlap_aconn_level0_jaccard = overlap_aconn_level0[:,sorterJ]

fig, ax = plt.subplots()
ax.imshow(rearanged_overlap_aconn_level0_jaccard, cmap=plt.cm.Blues)
for i in np.arange(len(groups_heirarchical_unique)):
    for j in np.arange(len(groups_heirarchical_unique_aconn)):
        c = rearanged_overlap_aconn_level0_jaccard[i,j]
        ax.text(j, i, str(int(c)), va='center', ha='center')
ax.set_ylabel("Block Fconn")
ax.set_xlabel("Block Aconn")
plt.xticks(ticks = np.arange(len(groups_heirarchical_unique_aconn)), labels = sorterJ+1)
plt.yticks(ticks = np.arange(len(groups_heirarchical_unique)), labels = np.arange(len(groups_heirarchical_unique))+1)
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_enrichment_jaccardsorted_mcmc.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/Aconn_level0_enrichment_jaccardsorted_mcmc.eps",
    dpi=300, bbox_inches="tight", format="eps")
plt.show()

group_exp_perm_transpose_jaccard = group_exp_permutations.transpose()[:,sorterJ]

plt.figure(figsize=(15, 7))
violin_parts = plt.violinplot(group_exp_perm_transpose_jaccard, showmeans = True, showextrema = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('black')
    pc.set_edgecolor('black')
# Make all the violin statistics marks red:
#for partname in ('cbars','cmins','cmaxes', "cmeans"):
# for partname in ("cmeans"):
vp = violin_parts["cmeans"]
vp.set_edgecolor("black")
vp.set_linewidth(1)
plt.plot([1],overlap_aconn_level0[group_exp,sorterJ][0], marker = 'o', markersize = 20, linestyle = "", markeredgewidth = 4, color = "orangered")
plt.plot(np.arange(19)+2,overlap_aconn_level0[group_exp,sorterJ][1:20], marker = 'o', markersize = 20, linestyle = "", markeredgewidth = 4, color = "cornflowerblue")
plt.ylabel('Overlap with Community '+ str(group_exp+1), fontsize = 30)
plt.xlabel('Anatomical Community ', fontsize = 30)
plt.xticks(ticks = np.arange(len(groups_heirarchical_unique_aconn))+1, labels = sorterJ+1, fontsize = 25)
plt.yticks([0,5,10,15], [0,5,10,15],fontsize = 25 )
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/aconn_enrichemnt_exp_violin_jaccard.pdf", dpi=300, bbox_inches="tight")
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/aconn_enrichemnt_exp_violin_jaccard.pdf", dpi=300, bbox_inches="tight")

plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/aconn_enrichemnt_exp_violin_jaccard.eps",
        dpi=300, bbox_inches="tight", format ="eps")
plt.show()


#### looking at scatterplots of how similar the aconn and funconn blocks are

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


overlap_SIM_fconn = np.zeros((len(groups_heirarchical_unique),3))
overlap_SIM_aconn = np.zeros((len(groups_heirarchical_unique_aconn),3))
for i in np.arange(len(groups_heirarchical_unique)):
    indicies_of_group = np.where(groups_heirarchical == groups_heirarchical_unique[i])[0]
    neurons_in_group = np.array(neuronlist)[indicies_of_group]
    for neuron in neurons_in_group:
        if neuron in sensory:
            overlap_SIM_fconn[i,0] += 1
        elif neuron in inter:
            overlap_SIM_fconn[i, 1] += 1
        elif neuron in motor:
            overlap_SIM_fconn[i, 2] += 1
for i in np.arange(len(groups_heirarchical_unique_aconn)):
    indicies_of_group = np.where(groups_heirarchical_aconn == groups_heirarchical_unique_aconn[i])[0]
    neurons_in_group = np.array(neuronlist)[indicies_of_group]
    for neuron in neurons_in_group:
        if neuron in sensory:
            overlap_SIM_aconn[i,0] += 1
        elif neuron in inter:
            overlap_SIM_aconn[i, 1] += 1
        elif neuron in motor:
            overlap_SIM_aconn[i, 2] += 1

Aconn_jaccard_indicies = sorterJ[:len(groups_heirarchical_unique)]
number_aconn = np.sum(overlap_aconn_level0, axis = 0)
number_fconn = np.sum(overlap_aconn_level0, axis = 1)
number_aconn_jaccard = number_aconn[sorterJ]
number_aconn_jaccard = number_aconn_jaccard[:len(groups_heirarchical_unique)]
plt.figure()
colors = ["#98CE87", "#FCBA77", "#F57F20", "#1377B5", "#ACC7E8", "#1DA149"]
symbols = ["*", "o", "D"]
labels = ["Sensory neurons", "Interneurons", "Motorneurons"]
x = []
y = []
for k in np.arange(len(groups_heirarchical_unique)):
    for sim in np.arange(3):
        plt.plot(overlap_SIM_aconn[Aconn_jaccard_indicies[k],sim]/number_aconn_jaccard[k],overlap_SIM_fconn[k,sim]/number_fconn[k],
                 color = colors[k], marker = symbols[sim], markersize = 15, label = labels[sim], linestyle = "")
        x.append(overlap_SIM_aconn[Aconn_jaccard_indicies[k],sim]/number_aconn_jaccard[k])
        y.append(overlap_SIM_fconn[k,sim]/number_fconn[k])
plt.plot([0,1],[0,1], "-", color= "black")
plt.xticks([0,0.5,1], [0,0.5,1], fontsize = 20)
plt.yticks([0,0.5,1], [0,0.5,1], fontsize = 20)
plt.xlabel("Fraction Neurontype Anatomical", fontsize = 15)
plt.ylabel("Fraction Neurontype Signal Propagation", fontsize = 15)
#plt.legend()
plt.tight_layout()
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/Aconn_level0_scatter_SIM.eps",
        dpi=300, bbox_inches="tight", format="eps")
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_scatter_SIM.pdf", dpi=300, bbox_inches="tight")
plt.show()

regr = linregress(x,y)
R = regr.rvalue
corrcoeff = np.corrcoef(x,y)
x_regr = [0,0.5,1]
y_regr = regr.slope*np.array(x_regr)+regr.intercept


plt.figure()
for k in np.arange(len(groups_heirarchical_unique)):
    for sim in np.arange(3):
        plt.plot(overlap_SIM_aconn[Aconn_jaccard_indicies[k],sim]/number_aconn_jaccard[k],overlap_SIM_fconn[k,sim]/number_fconn[k],
                 color = colors[k], marker = symbols[sim], markersize = 15, label = labels[sim], linestyle = "")
plt.plot(x_regr,y_regr, "-", color= "black")
plt.plot([0,1],[0,1], linestyle='dashed', color= "black")
plt.xticks([0,0.5,1], [0,0.5,1], fontsize = 20)
plt.yticks([0,0.5,1], [0,0.5,1], fontsize = 20)
plt.xlabel("Fraction Neurontype Anatomical", fontsize = 15)
plt.ylabel("Fraction Neurontype Signal Propagation", fontsize = 15)
#plt.legend()
plt.tight_layout()
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/Aconn_level0_scatter_SIM_regr_both.eps",
        dpi=300, bbox_inches="tight", format="eps")
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_scatter_SIM_regr_both.pdf", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
for k in np.arange(len(groups_heirarchical_unique)):
    for sim in np.arange(3):
        plt.plot(overlap_SIM_aconn[Aconn_jaccard_indicies[k],sim]/number_aconn_jaccard[k],overlap_SIM_fconn[k,sim]/number_fconn[k],
                 color = colors[k], marker = symbols[sim], markersize = 15, label = labels[sim], linestyle = "")
        # plt.plot(overlap_SIM_aconn[Aconn_jaccard_indicies, sim] / number_aconn_jaccard,
        #          overlap_SIM_fconn[:, sim] / number_fconn,
        #          color=colors[sim], marker=symbols[k], markersize=15, label=labels[sim], linestyle="")
plt.plot(x_regr,y_regr, "-", color= "black")
plt.xticks([0,0.5,1], [0,0.5,1], fontsize = 20)
plt.yticks([0,0.5,1], [0,0.5,1], fontsize = 20)
plt.xlabel("Fraction Neurontype Anatomical", fontsize = 15)
plt.ylabel("Fraction Neurontype Signal Propagation", fontsize = 15)
#plt.legend()
plt.tight_layout()
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/Aconn_level0_scatter_SIM_regr.eps",
        dpi=300, bbox_inches="tight", format="eps")
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_scatter_SIM_regr.pdf", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
for k in np.arange(len(groups_heirarchical_unique)):
    plt.plot(rearanged_overlap_aconn_level0_jaccard[k,k]/number_aconn_jaccard[k], rearanged_overlap_aconn_level0_jaccard[k,k]/number_fconn[k],
             markersize = 15, label = "Block"+str(k+1), linestyle = "")
plt.xlabel("Fraction Neurons Anatomical", fontsize = 15)
plt.ylabel("Fraction Neurons Signal Propagation", fontsize = 15)
plt.legend()
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_scatter_overlap.pdf", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
for k in np.arange(len(groups_heirarchical_unique)):
    plt.plot(number_aconn_jaccard[k], number_fconn[k],
             markersize = 15, marker = "o", label = "Community"+str(k+1), linestyle = "",
             color = colors[k])
plt.xticks([10,20,30], [10,20,30], fontsize = 20)
plt.yticks([10,20,30,40,50], [10,20,30,40,50], fontsize = 20)
plt.hist(number_aconn, color = "grey")
plt.xlabel("Nr of Neurons in Block Anatomical", fontsize = 15)
plt.ylabel("Nr of Neurons in Block Signal Propagation", fontsize = 15)
plt.legend()
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_scatter_nrneurons.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/Aconn_level0_scatter_nrneurons.eps",
        dpi=300, bbox_inches="tight", format="eps")
plt.show()


fig, ax1 = plt.subplots(figsize=(8, 6))

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
for k in np.arange(len(groups_heirarchical_unique)):
    ax1.plot(number_aconn_jaccard[k], number_fconn[k],
             markersize = 15, marker = "o", label = "Community"+str(k+1), linestyle = "",
             color = colors[k])

ax2.hist(number_aconn, bins = 20, color = "grey", alpha = 0.3, density = False)
ax1.set_xlabel("Nr of Neurons in Anatomical Community", fontsize = 15)
ax1.set_xticks([0,10,20],  labels=[0,10,20], fontsize = 20)
ax1.set_ylabel("Nr of Neurons in Signaling Community", fontsize = 15)
ax1.set_yticks([10,20,30,40,50], labels=[10,20,30,40,50], fontsize = 20)
ax2.set_ylabel("Nr of Anatomical Communities", fontsize = 15)
ax2.set_yticks([0,1,2], labels=[0,1,2], fontsize = 20)
ax1.legend()
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconn_level0_scatter_nrneurons_withhist.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/Aconn_level0_scatter_nrneurons_withhist.eps",
        dpi=300, bbox_inches="tight", format="eps")

plt.show()

print("done")