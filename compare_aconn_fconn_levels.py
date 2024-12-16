import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy import stats



## Comparing the different heirarchical levels

with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/get_adj_matrix.pkl', 'rb') as f:
    get_adj_matrix = pickle.load(f)
with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/get_adj_matrix_aconn.pkl', 'rb') as f:
    get_adj_matrix_aconn = pickle.load(f)
with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/neuronlist.pkl', 'rb') as g:
    neuronlist = pickle.load(g)


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


groups_heirarchical_all_fconn = get_sbm_stats(state, 188)


nr_of_levels_fconn = np.size(groups_heirarchical_all_fconn[0], axis=1)
nr_of_blocks_fconn = []
for i in np.arange(nr_of_levels_fconn):
    nr_of_blocks_fconn.append(len(np.unique(groups_heirarchical_all_fconn[0][:,i])))

print("The number of signal propagation hierarchies is "+str(len(np.unique(nr_of_blocks_fconn))))

groups_heirarchical_fconn_unique = groups_heirarchical_all_fconn[0][:,0:3]

groups_heirarchical_all_aconn = get_sbm_stats(state_aconn, 188)

nr_of_levels_aconn = np.size(groups_heirarchical_all_aconn[0], axis=1)
nr_of_blocks_aconn = []
for i in np.arange(nr_of_levels_aconn):
    nr_of_blocks_aconn.append(len(np.unique(groups_heirarchical_all_aconn[0][:,i])))

print("The number of signal propagation hierarchies is "+str(len(np.unique(nr_of_blocks_aconn))))


groups_heirarchical_aconn_unique = groups_heirarchical_all_aconn[0][:,0:5]


MI = np.empty((len(np.unique(nr_of_blocks_aconn)),len(np.unique(nr_of_blocks_fconn))))
ARS = np.empty((len(np.unique(nr_of_blocks_aconn)),len(np.unique(nr_of_blocks_fconn))))
MI.fill(np.NAN)
ARS.fill(np.NAN)
for a in np.arange(len(np.unique(nr_of_blocks_aconn))):
    for f in np.arange(len(np.unique(nr_of_blocks_fconn))):
        MI[a,f] = normalized_mutual_info_score(groups_heirarchical_aconn_unique[:,a], groups_heirarchical_fconn_unique[:,f])
        ARS[a,f] = adjusted_rand_score(groups_heirarchical_aconn_unique[:,a], groups_heirarchical_fconn_unique[:,f])


MI_rand = np.empty((100,len(np.unique(nr_of_blocks_aconn)),len(np.unique(nr_of_blocks_fconn))))
ARS_rand = np.empty((100,len(np.unique(nr_of_blocks_aconn)),len(np.unique(nr_of_blocks_fconn))))
MI_rand.fill(np.NAN)
ARS_rand.fill(np.NAN)
for n in np.arange(100):
    groups_heirarchical_aconn_unique_copy = np.copy(groups_heirarchical_aconn_unique)
    groups_heirarchical_fconn_unique_copy = np.copy(groups_heirarchical_fconn_unique)
    np.random.shuffle(groups_heirarchical_aconn_unique_copy)
    np.random.shuffle(groups_heirarchical_fconn_unique_copy)
    for a in np.arange(len(np.unique(nr_of_blocks_aconn))):
        for f in np.arange(len(np.unique(nr_of_blocks_fconn))):
            MI_rand[n, a, f] = normalized_mutual_info_score(groups_heirarchical_aconn_unique_copy[:, a],
                                                    groups_heirarchical_fconn_unique_copy[:, f])
            ARS_rand[n, a, f] = adjusted_rand_score(groups_heirarchical_aconn_unique_copy[:, a],
                                            groups_heirarchical_fconn_unique_copy[:, f])


MI_rand_ave = np.mean(MI_rand, axis = 0)
ARS_rand_ave = np.mean(ARS_rand, axis = 0)

MI_rand_std = np.std(MI_rand, axis = 0)
ARS_rand_std = np.std(ARS_rand, axis = 0)

MI_zscore = (MI - MI_rand_ave)/MI_rand_std
ARS_zscore = (ARS - ARS_rand_ave)/ARS_rand_std


MI_pval = np.empty((len(np.unique(nr_of_blocks_aconn)),len(np.unique(nr_of_blocks_fconn))))
ARS_pval = np.empty((len(np.unique(nr_of_blocks_aconn)),len(np.unique(nr_of_blocks_fconn))))
MI_pval.fill(np.NAN)
ARS_pval.fill(np.NAN)
for a in np.arange(len(np.unique(nr_of_blocks_aconn))):
    for f in np.arange(len(np.unique(nr_of_blocks_fconn))):
        MI_pval[a,f] = stats.ttest_ind(MI[a,f], MI_rand[:,a,f], alternative = "greater").pvalue
        ARS_pval[a, f] = stats.ttest_ind(ARS[a, f], ARS_rand[:, a, f], alternative="greater").pvalue

MI_pval_flattened = MI_pval[0:4,0:2].flatten()
ARS_pval_flattened = ARS_pval[0:4,0:2].flatten()

MI_pval_corrected = stats.false_discovery_control(MI_pval_flattened)
MI_pval_corrected_matrix = MI_pval_corrected.reshape(4,2)

ARS_pval_corrected = stats.false_discovery_control(ARS_pval_flattened)
ARS_pval_corrected_matrix = ARS_pval_corrected.reshape(4,2)

plt.figure()
plt.imshow(MI, cmap="Blues", vmin = 0, vmax = 1)
plt.tight_layout()
plt.xlabel("Signaling Levels")
plt.ylabel("Anatomical Levels")
cbar = plt.colorbar()
cbar.set_label('Normalized Mutual Information', rotation=270)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/compare_levels_aconn_fconn.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig3/compare_levels_aconn_fconn.eps",
    dpi=300, bbox_inches="tight", format="eps")
plt.show()



print("done")