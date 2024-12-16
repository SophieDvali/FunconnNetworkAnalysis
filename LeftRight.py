import graph_tool.all as gt
import numpy as np
import pandas as pd
import pickle
from scipy import stats

###Checking community assignment of bilaterally symmetric pairs in signaling and anatomy

#importing the matricies, states and making the graph:

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
levels_aconn = state_aconn.get_levels()

groups_heirarchical = []
for i in np.arange(len(neuronlist)):
    groups_heirarchical.append(levels[0].get_blocks()[i])

#Sorting the communities by size so that they are standardized across scripts
group_lengths = []
for gr in np.unique(groups_heirarchical):
    group_lengths.append(len(np.where(groups_heirarchical == gr)[0]))

group_sorter = np.argsort(group_lengths)

groups_heirarchical_unique = np.unique(groups_heirarchical)[group_sorter]

#For the signaling network

#calculating the number of bilateral pairs in the same community and in different communities

LR_pairs_same_block = 0
LR_pairs_not_same_block = 0

for ai in np.arange(188):
    for aj in np.arange(188):
        iid = neuronlist[ai]
        jid = neuronlist[aj]
        if ai != aj:
            if iid.endswith("L"):
                if jid == iid[0:-1] + "R" and groups_heirarchical[ai] == groups_heirarchical[aj]:
                    LR_pairs_same_block += 1
                elif jid == iid[0:-1] + "R" and groups_heirarchical[ai] != groups_heirarchical[aj]:
                    LR_pairs_not_same_block += 1
                    print(jid)
            elif iid.endswith("R"):
                if jid == iid[0:-1] + "L" and groups_heirarchical[ai] == groups_heirarchical[aj]:
                    LR_pairs_same_block += 1
                elif jid == iid[0:-1] + "L" and groups_heirarchical[ai] != groups_heirarchical[aj]:
                    LR_pairs_not_same_block += 1
                    print(jid)

Percent_LR_same = LR_pairs_same_block/(LR_pairs_same_block+LR_pairs_not_same_block)
Percent_LR_notsame = LR_pairs_not_same_block/(LR_pairs_same_block+LR_pairs_not_same_block)
print("for signaling" + str(Percent_LR_same/Percent_LR_notsame))

## for 100 random community assigments claculate the same
Percents_LR_same_rand = []
Percents_LR_not_same_rand = []

for n in np.arange(100):
    groups_heirarchical_rand =  np.copy(groups_heirarchical)
    np.random.shuffle(groups_heirarchical_rand)

    LR_pairs_same_block = 0
    LR_pairs_not_same_block = 0

    for ai in np.arange(188):
        for aj in np.arange(188):
            iid = neuronlist[ai]
            jid = neuronlist[aj]
            if ai != aj:
                if iid.endswith("L"):
                    if jid == iid[0:-1] + "R" and groups_heirarchical_rand[ai] == groups_heirarchical_rand[aj]:
                        LR_pairs_same_block += 1
                    elif jid == iid[0:-1] + "R" and groups_heirarchical_rand[ai] != groups_heirarchical_rand[aj]:
                        LR_pairs_not_same_block += 1
                        print(jid)
                elif iid.endswith("R"):
                    if jid == iid[0:-1] + "L" and groups_heirarchical_rand[ai] == groups_heirarchical_rand[aj]:
                        LR_pairs_same_block += 1
                    elif jid == iid[0:-1] + "L" and groups_heirarchical_rand[ai] != groups_heirarchical_rand[aj]:
                        LR_pairs_not_same_block += 1
                        print(jid)

    Percents_LR_same_rand.append(LR_pairs_same_block / (LR_pairs_same_block + LR_pairs_not_same_block))
    Percents_LR_not_same_rand.append(LR_pairs_not_same_block / (LR_pairs_same_block + LR_pairs_not_same_block))


pval = stats.ttest_ind(Percent_LR_same/Percent_LR_notsame, np.array(Percents_LR_same_rand)/np.array(Percents_LR_not_same_rand), alternative = "greater").pvalue

#aconn

levels_aconn = state_aconn.get_levels()

groups_heirarchical_aconn = []
for i in np.arange(len(neuronlist)):
    groups_heirarchical_aconn.append(levels_aconn[0].get_blocks()[i])

group_lengths_aconn = []
for gr in np.unique(groups_heirarchical_aconn):
    group_lengths_aconn.append(len(np.where(groups_heirarchical_aconn == gr)[0]))

group_sorter_aconn = np.argsort(group_lengths_aconn)

groups_heirarchical_unique_aconn = np.unique(groups_heirarchical_aconn)[group_sorter_aconn]


LR_pairs_same_block = 0
LR_pairs_not_same_block = 0

for ai in np.arange(188):
    for aj in np.arange(188):
        iid = neuronlist[ai]
        jid = neuronlist[aj]
        if ai != aj:

            if iid.endswith("L"):
                if jid == iid[0:-1] + "R" and groups_heirarchical_aconn[ai] == groups_heirarchical_aconn[aj]:
                    LR_pairs_same_block += 1
                elif jid == iid[0:-1] + "R" and groups_heirarchical_aconn[ai] != groups_heirarchical_aconn[aj]:
                    LR_pairs_not_same_block += 1
                    print(jid)
            elif iid.endswith("R"):
                if jid == iid[0:-1] + "L" and groups_heirarchical_aconn[ai] == groups_heirarchical_aconn[aj]:
                    LR_pairs_same_block += 1
                elif jid == iid[0:-1] + "L" and groups_heirarchical_aconn[ai] != groups_heirarchical_aconn[aj]:
                    LR_pairs_not_same_block += 1
                    print(jid)

Percent_LR_same_aconn = LR_pairs_same_block/(LR_pairs_same_block+LR_pairs_not_same_block)
Percent_LR_notsame_aconn = LR_pairs_not_same_block/(LR_pairs_same_block+LR_pairs_not_same_block)
print("for anatomy" +  str(Percent_LR_same_aconn/Percent_LR_notsame_aconn))


Percents_LR_same_rand_aconn = []
Percents_LR_not_same_rand_aconn = []

for n in np.arange(100):
    groups_heirarchical_rand_aconn =  np.copy(groups_heirarchical_aconn)
    np.random.shuffle(groups_heirarchical_rand_aconn)

    LR_pairs_same_block = 0
    LR_pairs_not_same_block = 0

    for ai in np.arange(188):
        for aj in np.arange(188):
            iid = neuronlist[ai]
            jid = neuronlist[aj]
            if ai != aj:

                if iid.endswith("L"):
                    if jid == iid[0:-1] + "R" and groups_heirarchical_rand_aconn[ai] == groups_heirarchical_rand_aconn[aj]:
                        LR_pairs_same_block += 1
                    elif jid == iid[0:-1] + "R" and groups_heirarchical_rand_aconn[ai] != groups_heirarchical_rand_aconn[aj]:
                        LR_pairs_not_same_block += 1
                        print(jid)
                elif iid.endswith("R"):
                    if jid == iid[0:-1] + "L" and groups_heirarchical_rand_aconn[ai] == groups_heirarchical_rand_aconn[aj]:
                        LR_pairs_same_block += 1
                    elif jid == iid[0:-1] + "L" and groups_heirarchical_rand_aconn[ai] != groups_heirarchical_rand_aconn[aj]:
                        LR_pairs_not_same_block += 1
                        print(jid)

    Percents_LR_same_rand_aconn.append(LR_pairs_same_block / (LR_pairs_same_block + LR_pairs_not_same_block))
    Percents_LR_not_same_rand_aconn.append(LR_pairs_not_same_block / (LR_pairs_same_block + LR_pairs_not_same_block))


pval_aconn = stats.ttest_ind(Percent_LR_same_aconn/Percent_LR_notsame_aconn, np.array(Percents_LR_same_rand_aconn)/np.array(Percents_LR_not_same_rand_aconn), alternative = "greater").pvalue

print("done")