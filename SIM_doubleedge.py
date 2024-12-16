import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import signal
import scipy as sp
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
import random
import sys
import matplotlib
from matplotlib_venn import venn2


with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/get_adj_matrix.pkl', 'rb') as f:
    get_adj_matrix = pickle.load(f)
with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/get_adj_matrix_aconn.pkl', 'rb') as f:
    get_adj_matrix_aconn = pickle.load(f)
with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/neuronlist.pkl', 'rb') as g:
    neuronlist = pickle.load(g)


df_q_head_binary = pd.read_csv('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/q_head_binary.csv')
q_head_binary = df_q_head_binary.to_numpy()
q_head_binary = np.delete(q_head_binary, obj=0, axis=1)


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

G = nx.DiGraph(get_adj_matrix)
G_aconn = nx.DiGraph(get_adj_matrix_aconn)

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





#I need the number of edges between S and I and M in all configurations

S_to_I = 0
S_to_S = 0
S_to_M = 0
S_to_all = 0

I_to_S = 0
I_to_I = 0
I_to_M = 0
I_to_all = 0

M_to_M = 0
M_to_I = 0
M_to_S = 0
M_to_all = 0


all_edges = [e for e in G.edges]

for edg in all_edges:
    source = edg[0]
    target = edg[1]
    source_neuron = neuronlist[source]
    target_neuron = neuronlist[target]

    if source_neuron in sensory:
        if target_neuron in inter:
            S_to_I += 1
            S_to_all += 1

        elif target_neuron in motor:
            S_to_M += 1
            S_to_all += 1

        elif target_neuron in sensory:
            S_to_S += 1
            S_to_all += 1

    elif source_neuron in inter:
        if target_neuron in motor:
            I_to_M += 1
            I_to_all += 1

        elif target_neuron in inter:
            I_to_I += 1
            I_to_all += 1

        elif target_neuron in sensory:
            I_to_S += 1
            I_to_all += 1

    elif source_neuron in motor:
        if target_neuron in sensory:
            M_to_S += 1
            M_to_all += 1

        elif target_neuron in inter:
            M_to_I += 1
            M_to_all += 1

        elif target_neuron in motor:
            M_to_M += 1
            M_to_all += 1

P_S_to_S = S_to_S/S_to_all
P_S_to_I = S_to_I/S_to_all
P_S_to_M = S_to_M/S_to_all
P_I_to_S = I_to_S/I_to_all
P_I_to_I = I_to_I/I_to_all
P_I_to_M = I_to_M/I_to_all
P_M_to_S = M_to_S/M_to_all
P_M_to_I = M_to_I/M_to_all
P_M_to_M = M_to_M/M_to_all


#defining a sort of double edge swap (modified from nx) where x->y and a->b get swapped to x->b and a->y
def swapping_edges(G, nswap=1, max_tries=2000, seed=None):
    n = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence
    while swapcount < nswap:
        #        if random.random() < 0.5: continue # trick to avoid periodicities?
        # pick two random edges without creating edge list
        # choose source node indices from discrete distribution
        (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=seed)
        random.seed(seed)
        if ui == xi:
            continue  # same source, skip
        u = keys[ui]  # convert index to label
        x = keys[xi]
        # choose target uniformly from neighbors
        if list(G[u]) and list(G[x]):
           v = random.choice(list(G[u]))
           y = random.choice(list(G[x]))
        else:
           continue

        if v == y:
            continue  # same target, skip
        if (y not in G[u]) and (v not in G[x]):  # don't create parallel edges
            G.add_edge(u, y)
            G.add_edge(x, v)
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            swapcount += 1
        if n >= max_tries:
            #e = (
            #    f"Maximum number of swap attempts ({n}) exceeded "
            #    f"before desired swaps achieved ({nswap})."
            #)
            print("Maximum number of swap attempts exceeded before desired swaps achieved.)")
            print(str(swapcount)+" swaps done")
            #raise nx.NetworkXAlgorithmError(e)
            return G
        n += 1
    return G

#compare to a random reshuffle of the network using the double edge swaps
nr_sensory = len(sensory)
nr_inter = len(inter)
nr_motor = len(motor)

P_S_to_S_rand = []
P_S_to_I_rand = []
P_S_to_M_rand = []
P_I_to_S_rand = []
P_I_to_I_rand = []
P_I_to_M_rand = []
P_M_to_S_rand = []
P_M_to_I_rand = []
P_M_to_M_rand = []
IM_frac = []
SI_frac = []
SM_frac = []
neuronlist_rand = neuronlist.copy()
for i in np.arange(1000):
    G_rand = G.copy()
    swapping_edges(G_rand, nswap = 2000, max_tries=4000, seed = None)

    S_to_I_rand = 0
    S_to_S_rand = 0
    S_to_M_rand = 0
    S_to_all_rand = 0

    I_to_S_rand = 0
    I_to_I_rand = 0
    I_to_M_rand = 0
    I_to_all_rand = 0

    M_to_M_rand = 0
    M_to_I_rand = 0
    M_to_S_rand = 0
    M_to_all_rand = 0

    all_edges_rand = [e for e in G_rand.edges]

    for edg in all_edges_rand:
        source = edg[0]
        target = edg[1]
        source_neuron = neuronlist[source]
        target_neuron = neuronlist[target]

        if source_neuron in sensory:
            if target_neuron in inter:
                S_to_I_rand += 1
                S_to_all_rand += 1

            elif target_neuron in motor:
                S_to_M_rand += 1
                S_to_all_rand += 1

            elif target_neuron in sensory:
                S_to_S_rand += 1
                S_to_all_rand += 1

        elif source_neuron in inter:
            if target_neuron in motor:
                I_to_M_rand += 1
                I_to_all_rand += 1

            elif target_neuron in inter:
                I_to_I_rand += 1
                I_to_all_rand += 1

            elif target_neuron in sensory:
                I_to_S_rand += 1
                I_to_all_rand += 1

        elif source_neuron in motor:
            if target_neuron in sensory:
                M_to_S_rand += 1
                M_to_all_rand += 1

            elif target_neuron in inter:
                M_to_I_rand += 1
                M_to_all_rand += 1

            elif target_neuron in motor:
                M_to_M_rand += 1
                M_to_all_rand += 1

    P_S_to_S_rand.append(S_to_S_rand / S_to_all_rand)
    P_S_to_I_rand.append(S_to_I_rand / S_to_all_rand)
    P_S_to_M_rand.append(S_to_M_rand / S_to_all_rand)
    P_I_to_S_rand.append(I_to_S_rand / I_to_all_rand)
    P_I_to_I_rand.append(I_to_I_rand / I_to_all_rand)
    P_I_to_M_rand.append(I_to_M_rand / I_to_all_rand)
    P_M_to_S_rand.append(M_to_S_rand / M_to_all_rand)
    P_M_to_I_rand.append(M_to_I_rand / M_to_all_rand)
    P_M_to_M_rand.append(M_to_M_rand / M_to_all_rand)

    IM_frac.append(I_to_M_rand/M_to_I_rand)
    SI_frac.append(S_to_I_rand/I_to_S_rand)
    SM_frac.append(S_to_M_rand/M_to_S_rand)


log_SI = np.log(P_S_to_I/P_I_to_S)
logs_SI = np.log(np.array(P_S_to_I_rand)/np.array(P_I_to_S_rand))

log_IM = np.log(P_I_to_M/P_M_to_I)
logs_IM = np.log(np.array(P_I_to_M_rand)/np.array(P_M_to_I_rand))

log_SM = np.log(P_S_to_M/P_M_to_S)
logs_SM = np.log(np.array(P_S_to_M_rand)/np.array(P_M_to_S_rand))


pval_log_SI = stats.ttest_ind(log_SI, np.array(logs_SI), alternative = "two-sided").pvalue
pval_log_IM = stats.ttest_ind(log_IM, np.array(logs_IM), alternative = "two-sided").pvalue
pval_log_SM = stats.ttest_ind(log_SM, np.array(logs_SM), alternative = "two-sided").pvalue

pval_frac_SI = stats.ttest_ind(np.log(S_to_I/I_to_S), np.log(np.array(SI_frac)), alternative = "two-sided").pvalue
pval_frac_IM = stats.ttest_ind(np.log(I_to_M/M_to_I), np.log(np.array(IM_frac)), alternative = "two-sided").pvalue
pval_frac_SM = stats.ttest_ind(np.log(S_to_M/M_to_S), np.log(np.array(SM_frac)), alternative = "two-sided").pvalue



for_violinplot_log = np.zeros((len(logs_IM), 3))
for_violinplot_log[:,0] = logs_SI
for_violinplot_log[:,1] = logs_IM
for_violinplot_log[:,2] = logs_SM


for_violinplot_frac = np.zeros((len(IM_frac), 3))
for_violinplot_frac[:,0] = np.log(SI_frac)
for_violinplot_frac[:,1] = np.log(IM_frac)
for_violinplot_frac[:,2] = np.log(SM_frac)

plt.figure()
violin_parts = plt.violinplot(for_violinplot_log, showmeans = True)
for pc in violin_parts['bodies']:
    pc.set_facecolor('darkgrey')
    pc.set_edgecolor('darkgrey')
for partname in ('cbars','cmins','cmaxes', "cmeans"):
    vp = violin_parts[partname]
    vp.set_edgecolor("darkgrey")
    vp.set_linewidth(1)
plt.plot([1],log_SI, marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.plot([2],log_IM, marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.plot([3],log_SM, marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.axhline(0, color = "black", linestyle = "dashed")
plt.text(0.95, 0.7, "*", fontsize = 25)
plt.ylabel('log(P_{forwards}/P_{backwards})')
plt.xticks([1,2,3], ["Sensory -> Inter", "Inter -> Motor", "Sensory -> Motor"], fontsize = 15)
plt.yticks([-1,0,1], [-1,0,1],fontsize = 15 )
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Log_violinplot_SIM_des.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig6/Log_violinplot_SIM_des.pdf",
        dpi=300, bbox_inches="tight")
plt.show()


plt.figure()
violin_parts = plt.violinplot(for_violinplot_frac, showmeans = True)
for pc in violin_parts['bodies']:
    pc.set_facecolor('darkgrey')
    pc.set_edgecolor('darkgrey')
# Make all the violin statistics marks red:
for partname in ('cbars','cmins','cmaxes', "cmeans"):
    vp = violin_parts[partname]
    vp.set_edgecolor("darkgrey")
    vp.set_linewidth(1)
plt.plot([1],np.log(S_to_I/I_to_S), marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.plot([2],np.log(I_to_M/M_to_I), marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.plot([3],np.log(S_to_M/M_to_S), marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.text(0.92, 0.8, "*", fontsize = 25)
#plt.text(2.95, 0.8, "*", fontsize = 25)
plt.axhline(0, color = "black", linestyle = "dashed")
plt.ylabel('log(N forwards/ N backwards)', fontsize = 15)
plt.xticks([1,2,3], ["Sensory -> Inter", "Inter -> Motor","Sensory -> Motor"], fontsize = 15)
plt.yticks([-1.5,-1,-0.5,0,0.5,1,1.5], [-1.5,-1,-0.5,0,0.5,1,1.5],fontsize = 15 )
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Frac_violinplot_SIM_des.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig6/Frac_violinplot_SIM_des.pdf",
        dpi=300, bbox_inches="tight")
plt.show()

###Aconn

S_to_I_aconn = 0
S_to_S_aconn = 0
S_to_M_aconn = 0
S_to_all_aconn = 0

I_to_S_aconn = 0
I_to_I_aconn = 0
I_to_M_aconn = 0
I_to_all_aconn = 0

M_to_M_aconn = 0
M_to_I_aconn = 0
M_to_S_aconn = 0
M_to_all_aconn = 0


all_edges_aconn = [e for e in G_aconn.edges]

for edg in all_edges_aconn:
    source = edg[0]
    target = edg[1]
    source_neuron = neuronlist[source]
    target_neuron = neuronlist[target]

    if source_neuron in sensory:
        if target_neuron in inter:
            S_to_I_aconn += 1
            S_to_all_aconn += 1

        elif target_neuron in motor:
            S_to_M_aconn += 1
            S_to_all_aconn += 1

        elif target_neuron in sensory:
            S_to_S_aconn += 1
            S_to_all_aconn += 1

    elif source_neuron in inter:
        if target_neuron in motor:
            I_to_M_aconn += 1
            I_to_all_aconn += 1

        elif target_neuron in inter:
            I_to_I_aconn += 1
            I_to_all_aconn += 1

        elif target_neuron in sensory:
            I_to_S_aconn += 1
            I_to_all_aconn += 1

    elif source_neuron in motor:
        if target_neuron in sensory:
            M_to_S_aconn += 1
            M_to_all_aconn += 1

        elif target_neuron in inter:
            M_to_I_aconn += 1
            M_to_all_aconn += 1

        elif target_neuron in motor:
            M_to_M_aconn += 1
            M_to_all_aconn += 1


P_S_to_I_aconn = S_to_I_aconn/S_to_all_aconn
P_I_to_M_aconn = I_to_M_aconn/I_to_all_aconn
P_S_to_S_aconn = S_to_S_aconn/S_to_all_aconn

P_S_to_S_aconn = S_to_S_aconn/S_to_all_aconn
P_S_to_I_aconn = S_to_I_aconn/S_to_all_aconn
P_S_to_M_aconn = S_to_M_aconn/S_to_all_aconn
P_I_to_S_aconn = I_to_S_aconn/I_to_all_aconn
P_I_to_I_aconn = I_to_I_aconn/I_to_all_aconn
P_I_to_M_aconn = I_to_M_aconn/I_to_all_aconn
P_M_to_S_aconn = M_to_S_aconn/M_to_all_aconn
P_M_to_I_aconn = M_to_I_aconn/M_to_all_aconn
P_M_to_M_aconn = M_to_M_aconn/M_to_all_aconn

P_S_to_S_rand_aconn = []
P_S_to_I_rand_aconn = []
P_S_to_M_rand_aconn = []
P_I_to_S_rand_aconn = []
P_I_to_I_rand_aconn = []
P_I_to_M_rand_aconn = []
P_M_to_S_rand_aconn = []
P_M_to_I_rand_aconn = []
P_M_to_M_rand_aconn = []
IM_frac_aconn = []
SI_frac_aconn = []
SM_frac_aconn = []
neuronlist_rand = neuronlist.copy()
for i in np.arange(1000):
    G_rand_aconn = G_aconn.copy()
    swapping_edges(G_rand_aconn, nswap = 6000, max_tries=9000, seed = None)

    S_to_I_rand_aconn = 0
    S_to_S_rand_aconn = 0
    S_to_M_rand_aconn = 0
    S_to_all_rand_aconn = 0

    I_to_S_rand_aconn = 0
    I_to_I_rand_aconn = 0
    I_to_M_rand_aconn = 0
    I_to_all_rand_aconn = 0

    M_to_M_rand_aconn = 0
    M_to_I_rand_aconn = 0
    M_to_S_rand_aconn = 0
    M_to_all_rand_aconn = 0

    all_edges_rand_aconn = [e for e in G_rand_aconn.edges]

    for edg in all_edges_rand_aconn:
        source = edg[0]
        target = edg[1]
        source_neuron = neuronlist[source]
        target_neuron = neuronlist[target]

        if source_neuron in sensory:
            if target_neuron in inter:
                S_to_I_rand_aconn += 1
                S_to_all_rand_aconn += 1

            elif target_neuron in motor:
                S_to_M_rand_aconn += 1
                S_to_all_rand_aconn += 1

            elif target_neuron in sensory:
                S_to_S_rand_aconn += 1
                S_to_all_rand_aconn += 1

        elif source_neuron in inter:
            if target_neuron in motor:
                I_to_M_rand_aconn += 1
                I_to_all_rand_aconn += 1

            elif target_neuron in inter:
                I_to_I_rand_aconn += 1
                I_to_all_rand_aconn += 1

            elif target_neuron in sensory:
                I_to_S_rand_aconn += 1
                I_to_all_rand_aconn += 1

        elif source_neuron in motor:
            if target_neuron in sensory:
                M_to_S_rand_aconn += 1
                M_to_all_rand_aconn += 1

            elif target_neuron in inter:
                M_to_I_rand_aconn += 1
                M_to_all_rand_aconn += 1

            elif target_neuron in motor:
                M_to_M_rand_aconn += 1
                M_to_all_rand_aconn += 1

    P_S_to_S_rand_aconn.append(S_to_S_rand_aconn / S_to_all_rand_aconn)
    P_S_to_I_rand_aconn.append(S_to_I_rand_aconn / S_to_all_rand_aconn)
    P_S_to_M_rand_aconn.append(S_to_M_rand_aconn / S_to_all_rand_aconn)
    P_I_to_S_rand_aconn.append(I_to_S_rand_aconn / I_to_all_rand_aconn)
    P_I_to_I_rand_aconn.append(I_to_I_rand_aconn / I_to_all_rand_aconn)
    P_I_to_M_rand_aconn.append(I_to_M_rand_aconn / I_to_all_rand_aconn)
    P_M_to_S_rand_aconn.append(M_to_S_rand_aconn / M_to_all_rand_aconn)
    P_M_to_I_rand_aconn.append(M_to_I_rand_aconn / M_to_all_rand_aconn)
    P_M_to_M_rand_aconn.append(M_to_M_rand_aconn / M_to_all_rand_aconn)

    IM_frac_aconn.append(I_to_M_rand_aconn/M_to_I_rand_aconn)
    SI_frac_aconn.append(S_to_I_rand_aconn/I_to_S_rand_aconn)
    SM_frac_aconn.append(S_to_M_rand_aconn/M_to_S_rand_aconn)




log_SI_aconn = np.log(P_S_to_I_aconn/P_I_to_S_aconn)
logs_SI_aconn = np.log(np.array(P_S_to_I_rand_aconn)/np.array(P_I_to_S_rand_aconn))

log_IM_aconn = np.log(P_I_to_M_aconn/P_M_to_I_aconn)
logs_IM_aconn = np.log(np.array(P_I_to_M_rand_aconn)/np.array(P_M_to_I_rand_aconn))

log_SM_aconn = np.log(P_S_to_M_aconn/P_M_to_S_aconn)
logs_SM_aconn = np.log(np.array(P_S_to_M_rand_aconn)/np.array(P_M_to_S_rand_aconn))


pval_log_SI_aconn = stats.ttest_ind(log_SI_aconn, np.array(logs_SI_aconn), alternative = "two-sided").pvalue
pval_log_IM_aconn = stats.ttest_ind(log_IM_aconn, np.array(logs_IM_aconn), alternative = "two-sided").pvalue
pval_log_SM_aconn = stats.ttest_ind(log_SM_aconn, np.array(logs_SM_aconn), alternative = "two-sided").pvalue

pval_frac_SI_aconn = stats.ttest_ind(np.log(S_to_I_aconn/I_to_S_aconn), np.log(np.array(SI_frac_aconn)), alternative = "two-sided").pvalue
pval_frac_IM_aconn = stats.ttest_ind(np.log(I_to_M_aconn/M_to_I_aconn), np.log(np.array(IM_frac_aconn)), alternative = "two-sided").pvalue
pval_frac_SM_aconn = stats.ttest_ind(np.log(S_to_M_aconn/M_to_S_aconn), np.log(np.array(SM_frac_aconn)), alternative = "two-sided").pvalue


for_violinplot_log_aconn = np.zeros((len(logs_IM_aconn), 3))
for_violinplot_log_aconn[:,0] = logs_SI_aconn
for_violinplot_log_aconn[:,1] = logs_IM_aconn
for_violinplot_log_aconn[:,2] = logs_SM_aconn


for_violinplot_frac_aconn = np.zeros((len(IM_frac_aconn), 3))
for_violinplot_frac_aconn[:,0] = np.log(SI_frac_aconn)
for_violinplot_frac_aconn[:,1] = np.log(IM_frac_aconn)
for_violinplot_frac_aconn[:,2] = np.log(SM_frac_aconn)

plt.figure()
violin_parts = plt.violinplot(for_violinplot_frac_aconn, showmeans = True)
for pc in violin_parts['bodies']:
    pc.set_facecolor('darkgrey')
    pc.set_edgecolor('darkgrey')
# Make all the violin statistics marks red:
for partname in ('cbars','cmins','cmaxes', "cmeans"):
    vp = violin_parts[partname]
    vp.set_edgecolor("darkgrey")
    vp.set_linewidth(1)
plt.plot([1],np.log(S_to_I_aconn/I_to_S_aconn), marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.plot([2],np.log(I_to_M_aconn/M_to_I_aconn), marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.plot([3],np.log(S_to_M_aconn/M_to_S_aconn), marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.text(2.92, 1.35, "**", fontsize = 25)
plt.ylabel('log(N forwards/ N backwards)', fontsize = 15)
plt.axhline(0, color = "black", linestyle = "dashed")
plt.xticks([1,2,3], ["Sensory -> Inter", "Inter -> Motor", "Sensory -> Motor"], fontsize = 15)
plt.yticks([-1.5,-1,-0.5,0,0.5,1,1.5], [-1.5,-1,-0.5,0,0.5,1,1.5],fontsize = 15 )
plt.ylim(-1.7, 1.7)
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Frac_violinplot_SIM_aconn_des.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig6/Frac_violinplot_SIM_aconn_des.eps",
        dpi=300, bbox_inches="tight", format ="eps")
plt.show()


plt.figure()
violin_parts = plt.violinplot(for_violinplot_log_aconn, showmeans = True)
for pc in violin_parts['bodies']:
    pc.set_facecolor('darkgrey')
    pc.set_edgecolor('darkgrey')
# Make all the violin statistics marks red:
for partname in ('cbars','cmins','cmaxes', "cmeans"):
    vp = violin_parts[partname]
    vp.set_edgecolor("darkgrey")
    vp.set_linewidth(1)
plt.plot([1],log_SI_aconn, marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.plot([2],log_IM_aconn, marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.plot([3],log_SM_aconn, marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
#plt.text(0.92, 0.7, "**", fontsize = 25)
plt.text(2.92, 0.7, "**", fontsize = 25)
plt.ylabel('log(P_{forwards}/P_{backwards})', fontsize = 15)
plt.axhline(0, color = "black", linestyle = "dashed")
plt.xticks([1,2, 3], ["Sensory -> Inter", "Inter -> Motor", "Sensory -> Motor"], fontsize = 15)
plt.yticks([-1,0,1], [-1,0,1],fontsize = 15 )
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Log_violinplot_SIM_aconn_des.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig6/Log_violinplot_SIM_aconn_des.eps",
        dpi=300, bbox_inches="tight", format ="eps")
plt.show()


print("done")