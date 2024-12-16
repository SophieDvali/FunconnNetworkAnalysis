import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import scipy as sp
import random
import networkx as nx

### Checking if the pharynx and amphid neurons are broadcasters or integrators

### importing all the matricies/files

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

G = nx.DiGraph(get_adj_matrix)
G_aconn = nx.DiGraph(get_adj_matrix_aconn)

degree = []
indegree =[]
outdegree = []
degree_nx = []
indegree_nx =[]
outdegree_nx = []
vn = 0
for v in g.vertices():
    degree.append(v.out_degree() + v.in_degree())
    indegree.append(v.in_degree())
    outdegree.append(v.out_degree())
    indegree_nx.append(G.in_degree(vn))
    outdegree_nx.append(G.out_degree(vn))
    degree_nx.append(G.degree(vn))
    vn += 1

#import observations and stimulations
#observation_sorted/stimulations_sorted signifies the list of observations that has been sorted according to
#the number of observations
#obesrvations/stimulations are sorted according to the neuronlist alphabetically
df_observations = pd.read_csv('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/observations.csv')
neurons_observation_sorted = df_observations["neuron"].tolist()
observation_sorted = df_observations["observed"].to_numpy()
observations = np.array(observation_sorted)[np.argsort(neurons_observation_sorted)]
stimulations_sorted = df_observations["stimulated"].to_numpy()
stimulations = np.array(stimulations_sorted)[np.argsort(neurons_observation_sorted)]


Amphid = ["ADFL", "ADLL", "AFDL", "ASEL", "ASGL", "ASHL", "ASIL", "ASJL", "ASKL", "AWAL", "AWBL", "AWCL", "ADFR",
          "ADLR", "AFDR","ASER", "ASGR", "ASHR", "ASIR", "ASJR", "ASKR", "AWAR", "AWBR", "AWCR"]


pharynx = ["I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6", "M1", "M2L", "M2R",
           "M3L", "M3R", "M4", "M5", "MCL", "MCR", "MI", "NSML", "NSMR"]

#getting the neuronlist indicies for the pharynx and amphid neurons
indices_pharynx = [i for i, n in enumerate(neuronlist) if n in pharynx]
indices_amphid = [i for i, n in enumerate(neuronlist) if n in Amphid]


indegree_pharynx = np.array(indegree_nx)[indices_pharynx]
outdegree_pharynx = np.array(outdegree_nx)[indices_pharynx]

indegree_amphid = np.array(indegree_nx)[indices_amphid]
outdegree_amphid = np.array(outdegree_nx)[indices_amphid]

obs_pharynx = np.array(observations)[indices_pharynx]
stim_pharynx = np.array(stimulations)[indices_pharynx]

obs_amphid = np.array(observations)[indices_amphid]
stim_amphid = np.array(stimulations)[indices_amphid]

#slope1, intercept1, r_value1, p_value1, std_err1 = sp.stats.linregress(degree_aconn, degree)

slope, intercept, r_value, p_value, std_err = sp.stats.linregress(observation_sorted, stimulations_sorted)

#Scatterplots

figp1 = plt.figure()
plt.scatter(observation_sorted, stimulations_sorted, color = "grey", label = "All neurons")
plt.scatter(obs_pharynx, stim_pharynx, color = "red", label = "Pharynx neurons")
plt.xlabel("Observations", fontsize = 15)
plt.ylabel("Stimulations", fontsize = 15)
plt.title("Pharynx", fontsize = 15)
plt.xticks([0,50, 100], [0,50, 100], fontsize = 15)
plt.yticks([0,30, 60], [0,30, 60],fontsize = 15 )
plt.legend()
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Obs_stim_pharynx.pdf",
    dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

figa1 = plt.figure()
plt.scatter(observation_sorted, stimulations_sorted, color = "grey",  label = "All neurons")
plt.scatter(obs_amphid, stim_amphid, color = "red",  label = "Amphid neurons")
plt.xlabel("Observations", fontsize = 15)
plt.ylabel("Stimulations", fontsize = 15)
plt.xticks([0,50, 100], [0,50, 100], fontsize = 15)
plt.yticks([0,30, 60], [0,30, 60],fontsize = 15 )
plt.title("Amphid", fontsize = 15)
plt.legend()
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Obs_stim_amphid.pdf",
    dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()


#Histograms: Stim and Pharynx and Obs and Amphid
# figa2 = plt.figure()
# figa2.set_size_inches(figa1.get_size_inches()[0], figa1.get_size_inches()[1] / 3)
# plt.hist(observation_sorted, color = "grey", bins = 20, density = False, alpha = 0.5, label = "All neurons")
# plt.hist(obs_amphid, color = "red", bins = 20, density = False, alpha = 0.5, label = "Amphid neurons")
# plt.title("Amphid sensory neurons",fontsize = 15)
# plt.xlabel("Observations",fontsize = 15)
# plt.ylabel("Number",fontsize = 15)
# plt.xticks([0,50, 100], [0,50, 100], fontsize = 15)
# plt.yticks([0,20], [0,20],fontsize = 15 )
# plt.legend()
# plt.savefig(
#     "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Int_borad_amphid_obs_hist.pdf",
#     dpi=300, bbox_inches="tight")
# plt.tight_layout()
# plt.show()

figa2 = plt.figure()
figa2.set_size_inches(figa1.get_size_inches()[0], figa1.get_size_inches()[1] / 3)
plt.hist(observation_sorted, color = "grey", bins = 20, density = True, alpha = 0.5, label = "All neurons")
plt.hist(obs_amphid, color = "red", bins = 20, density = True, alpha = 0.5, label = "Amphid neurons")
plt.title("Amphid sensory neurons",fontsize = 15)
plt.xlabel("Observations",fontsize = 15)
plt.ylabel("Density",fontsize = 15)
plt.xticks([0,50, 100], [0,50, 100], fontsize = 15)
plt.yticks([0,0.03], [0,0.03],fontsize = 15 )
plt.legend()
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Int_borad_amphid_obs_hist_density.pdf",
    dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

#pharynx

# figp2 = plt.figure()
# figp2.set_size_inches(figp1.get_size_inches()[1], figp1.get_size_inches()[1] / 3)
# plt.hist(stimulations_sorted, color = "grey", bins = 20, density = False, alpha = 0.5, label = "All neurons")
# plt.hist(stim_pharynx, color = "red", bins = 20, density = False, alpha = 0.5, label = "Pharynx neurons")
# plt.title("Pharynx neurons",fontsize = 15)
# plt.xlabel("Stimulations",fontsize = 15)
# plt.ylabel("Number",fontsize = 15)
# plt.xticks([0,30, 60], [0,30, 60], fontsize = 15)
# plt.yticks([0,30], [0,30],fontsize = 15 )
# plt.legend()
# plt.savefig(
#     "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Int_borad_pharynx_stim_hist.pdf",
#     dpi=300, bbox_inches="tight")
# plt.tight_layout()
# plt.show()

figp2 = plt.figure()
figp2.set_size_inches(figp1.get_size_inches()[1], figp1.get_size_inches()[1] / 3)
plt.hist(stimulations_sorted, color = "grey", bins = 20, density = True, alpha = 0.5, label = "All neurons")
plt.hist(stim_pharynx, color = "red", bins = 20, density = True, alpha = 0.5, label = "Pharynx neurons")
plt.title("Pharynx neurons",fontsize = 15)
plt.xlabel("Stimulations",fontsize = 15)
plt.ylabel("Density",fontsize = 15)
plt.xticks([0,30, 60], [0,30, 60], fontsize = 15)
plt.yticks([0,0.06], [0,0.06],fontsize = 15 )
plt.legend()
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Int_borad_pharynx_stim_hist_density.pdf",
    dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()


#Kolmogorov Smirnoff test to compare pharynx and amphid to the distriburion from all other neurons

stimulations_sorted_to_remove_pharynx = np.copy(stimulations_sorted)
stimulations_sorted_without_pharynx = np.delete(stimulations_sorted_to_remove_pharynx, indices_pharynx)
observation_sorted_to_remove_amphid = np.copy(observation_sorted)
observation_sorted_without_amphid = np.delete(observation_sorted_to_remove_amphid, indices_amphid)

ks_statistic_pharynx, p_value_pharynx = sp.stats.ks_2samp(stimulations_sorted_without_pharynx, stim_pharynx)
ks_statistic_amphid, p_value_amphid = sp.stats.ks_2samp(observation_sorted_without_amphid, obs_amphid)

if p_value_pharynx < 0.05:
    print("Reject null hypothesis: The pharynx stim distribution is significantly different from all other neurons.")
else:
    print("Fail to reject null hypothesis: The pharynx stim distribution is not significantly different from all other neurons.")

if p_value_amphid < 0.05:
    print("Reject null hypothesis: The amphid obs distribution is significantly different from all other neurons.")
else:
    print("Fail to reject null hypothesis: The amphid obs distribution is not significantly different from all other neurons.")



# y_line = slope*observations+intercept
# y_line_ps = slope*np.arange(min(observations),max(observations),1)+intercept+(1*std_err)
# y_line_ms = slope*np.arange(min(observations),max(observations),1)+intercept-(1*std_err)

plt.figure()
#plt.scatter(observation_sorted, stimulations_sorted, color = "grey")
plt.plot(np.arange(max(observations)), (np.arange(max(observations))*slope) +intercept)
#plt.fill_between(np.arange(min(observations),max(observations),1), y_line_ms, y_line_ps, color = "grey", alpha = 0.3)
plt.scatter(obs_pharynx, stim_pharynx, color = "red")
plt.xlabel("Observations")
plt.ylabel("Stimulations")
plt.title("Pharynx")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Obs_stim_pharynx_line.pdf",
    dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

plt.figure()
#plt.scatter(observation_sorted, stimulations_sorted, color = "grey")
plt.plot(np.arange(max(observations)), (np.arange(max(observations))*slope) +intercept)
plt.scatter(obs_amphid, stim_amphid, color = "red")
plt.xlabel("Observations")
plt.ylabel("Stimulations")
plt.title("Amphid")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Obs_stim_amphid_line.pdf",
    dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

#claculating the edges in and out of the pharynx and amphid neurons

all_edges = [e for e in G.edges]
nr_outof_pharynx = 0
nr_into_pharynx = 0
nr_outof_amphid = 0
nr_into_amphid = 0
for edg in all_edges:
    source = edg[0]
    target = edg[1]
    source_neuron = neuronlist[source]
    target_neuron = neuronlist[target]
    if source_neuron in pharynx and target_neuron not in pharynx:
        nr_outof_pharynx += 1

    if source_neuron not in pharynx and target_neuron in pharynx:
        nr_into_pharynx += 1

    if source_neuron in Amphid and target_neuron not in Amphid:
        nr_outof_amphid += 1

    if source_neuron not in Amphid and target_neuron in Amphid:
        nr_into_amphid += 1

nr_in_vs_out_pharynx = nr_into_pharynx/nr_outof_pharynx
nr_in_vs_out_amphid = nr_into_amphid/nr_outof_amphid


#compared to 1000 random assignments of pharynx and amphid neurons

nr_in_vs_out_pharynx_rand = []
nr_in_vs_out_amphid_rand = []
for i in np.arange(1000):
    random_indices = random.sample(range(188), len(pharynx)+len(Amphid))
    pharynx_indices_rand = random_indices[:len(pharynx)]
    pharynx_neurons_rand = neuronlist[pharynx_indices_rand]
    amphid_indices_rand = random_indices[len(pharynx):]
    amphid_neurons_rand = neuronlist[amphid_indices_rand]

    nr_outof_pharynx_rand = 0
    nr_into_pharynx_rand = 0
    nr_outof_amphid_rand = 0
    nr_into_amphid_rand = 0
    for edg in all_edges:
        source = edg[0]
        target = edg[1]
        source_neuron = neuronlist[source]
        target_neuron = neuronlist[target]
        if source_neuron in pharynx_neurons_rand and target_neuron not in pharynx_neurons_rand:
            nr_outof_pharynx_rand += 1

        if source_neuron not in pharynx_neurons_rand and target_neuron in pharynx_neurons_rand:
            nr_into_pharynx_rand += 1

        if source_neuron in amphid_neurons_rand and target_neuron not in amphid_neurons_rand:
            nr_outof_amphid_rand += 1

        if source_neuron not in amphid_neurons_rand and target_neuron in amphid_neurons_rand:
            nr_into_amphid_rand += 1

    nr_in_vs_out_pharynx_rand.append(nr_into_pharynx_rand/nr_outof_pharynx_rand)
    nr_in_vs_out_amphid_rand.append(nr_into_amphid_rand/nr_outof_amphid_rand)


nr_in_vs_out_pharynx_rand_pvalue = stats.ttest_ind(nr_in_vs_out_pharynx, nr_in_vs_out_pharynx_rand,
                                                alternative="less").pvalue
nr_in_vs_out_amphid_rand_pvalue = stats.ttest_ind(nr_in_vs_out_amphid, nr_in_vs_out_amphid_rand,
                                                alternative="greater").pvalue
#histograms:
plt.figure()
plt.hist(nr_in_vs_out_amphid_rand, density = True, alpha = 0.7)
plt.axvline(nr_in_vs_out_amphid, color = "red", linewidth = 2)
plt.title("Amphid sensory neurons")
plt.xlabel("Number of edges in/ number of edges out")
plt.ylabel("Density")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Integrator_broadcaster_amphid.pdf",
    dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(nr_in_vs_out_pharynx_rand, density = True, alpha = 0.7)
plt.axvline(nr_in_vs_out_pharynx, color = "red", linewidth = 2)
plt.title("Pharynx neurons")
plt.xlabel("Number of edges in/ number of edges out")
plt.ylabel("Density")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Integrator_broadcaster_pharynx.pdf",
    dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()



## violinplots:
for_violinplot = np.zeros((len(nr_in_vs_out_pharynx_rand), 2))
for_violinplot[:,0] = np.log(nr_in_vs_out_pharynx_rand)
for_violinplot[:,1] = np.log(nr_in_vs_out_amphid_rand)


plt.figure()
violin_parts = plt.violinplot(for_violinplot, showmeans = True)
for pc in violin_parts['bodies']:
    pc.set_facecolor('darkgrey')
    pc.set_edgecolor('darkgrey')
for partname in ('cbars','cmins','cmaxes', "cmeans"):
    vp = violin_parts[partname]
    vp.set_edgecolor("darkgrey")
    vp.set_linewidth(1)
plt.plot([1],np.log(nr_in_vs_out_pharynx), marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.plot([2],np.log(nr_in_vs_out_amphid), marker = '_', markersize = 45, linestyle = "", markeredgewidth = 5, color = "red")
plt.axhline(0, color = "black", linestyle = "dashed")
plt.text(0.95, 0.9, "*", fontsize = 25)
plt.text(1.95, 0.9, "**", fontsize = 25)
plt.ylabel('log(Nr. edges in/Nr edges out)', fontsize = 15)
plt.xticks([1,2], ["Pharynx", "Amphid Sensory Neurons"], fontsize = 15)
plt.yticks([-1,0,1], [-1,0,1],fontsize = 15 )
plt.ylim(-1.2, 1.2)
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Integrator_broadcaster.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig4/Integrator_broadcaster.pdf",
        dpi=300, bbox_inches="tight")
plt.show()


print("done")