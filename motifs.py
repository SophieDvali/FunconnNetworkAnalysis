
import pandas as pd
import networkx as nx
import matplotlib as mpl
import graph_tool.all as gt
import numpy as np
import pickle

mpl.rcParams["pdf.fonttype"] = 42


df_observations = pd.read_csv('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/observations.csv')
neurons_observation_sorted = df_observations["neuron"].tolist()
observation_sorted = df_observations["observed"].to_numpy()
observations = np.array(observation_sorted)[np.argsort(neurons_observation_sorted)]


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

degree = []
indegree =[]
outdegree = []
for v in g.vertices():
    degree.append(v.out_degree() + v.in_degree())
    indegree.append(v.in_degree())
    outdegree.append(v.out_degree())


degree_aconn = []
indegree_aconn =[]
outdegree_aconn = []
for v in g_aconn.vertices():
    degree_aconn.append(v.out_degree() + v.in_degree())
    indegree_aconn.append(v.in_degree())
    outdegree_aconn.append(v.out_degree())



n = 188  # number of nodes

G = nx.DiGraph(get_adj_matrix)
G_aconn = nx.DiGraph(get_adj_matrix_aconn)
Reciprocity = nx.reciprocity(G)
clust_nx = nx.average_clustering(G)
clust_aconn_nx = nx.average_clustering(G_aconn)

bc = nx.betweenness_centrality(G)
ave_bc = np.mean(list(bc.values()))
bc_aconn = nx.betweenness_centrality(G_aconn)
ave_bc_aconn = np.mean(list(bc_aconn.values()))

shortest_path = nx.shortest_path_length(G)
average_shortest_path = np.mean([np.mean(list(j.values()))for (i,j) in nx.shortest_path_length(G)])
for (i,j) in shortest_path:
    paths = j.values()

shortest_path_aconn = nx.shortest_path_length(G_aconn)
average_shortest_path_aconn = np.mean([np.mean(list(j.values()))for (i,j) in nx.shortest_path_length(G_aconn)])
for (i,j) in shortest_path_aconn:
    paths = j.values()


clusts_ER = np.empty((1,100))
clusts_ER.fill(np.NaN)
clusts_ER_aconn = np.empty((1,100))
clusts_ER_aconn.fill(np.NaN)
shortest_paths_ER = np.empty((1,100))
shortest_paths_ER.fill(np.NaN)
shortest_paths_ER_aconn = np.empty((1,100))
shortest_paths_ER_aconn.fill(np.NaN)

for i in np.arange(100):
    G_ER = nx.erdos_renyi_graph(188, 0.033, directed=True)
    G_ER_aconn = nx.erdos_renyi_graph(188, 0.092, directed=True)

    clust_nx_ER = nx.average_clustering(G_ER)
    clust_aconn_nx_ER = nx.average_clustering(G_ER_aconn)
    clusts_ER[0][i] = clust_nx_ER
    clusts_ER_aconn[0][i] = clust_aconn_nx_ER

    shortest_path_ER = nx.shortest_path_length(G_ER)
    average_shortest_path_ER = np.mean([np.mean(list(j.values()))for (i,j) in nx.shortest_path_length(G_ER)])
    shortest_paths_ER[0][i] = average_shortest_path_ER
    shortest_path_aconn_ER = nx.shortest_path_length(G_ER_aconn)
    average_shortest_path_aconn_ER = np.mean([np.mean(list(j.values()))for (i,j) in nx.shortest_path_length(G_ER_aconn)])
    shortest_paths_ER_aconn[0][i] = average_shortest_path_aconn_ER

smallworld = (clust_nx/ np.mean(clusts_ER)) / (average_shortest_path/ np.mean(shortest_paths_ER))
smallworld_aconn = (clust_aconn_nx/ np.mean(clusts_ER_aconn)) / (average_shortest_path_aconn/ np.mean(shortest_paths_ER_aconn))

###three node motif structures

triads = nx.triads_by_type(G)

neurons_in_021D = []
for struc in np.arange(len(triads["021D"])):
    nodes_in_struc = triads["021D"][struc].nodes()
    neurons_in_021D.append(list(nodes_in_struc))

nr_min_one_021D = len(np.unique(neurons_in_021D))

neurons_in_021U = []
for struc in np.arange(len(triads["021U"])):
    nodes_in_struc = triads["021U"][struc].nodes()
    neurons_in_021U.append(list(nodes_in_struc))

nr_min_one_021U = len(np.unique(neurons_in_021U))

neurons_in_021C = []
for struc in np.arange(len(triads["021C"])):
    nodes_in_struc = triads["021C"][struc].nodes()
    neurons_in_021C.append(list(nodes_in_struc))

nr_min_one_021C = len(np.unique(neurons_in_021C))

neurons_in_030T = []
for struc in np.arange(len(triads["030T"])):
    nodes_in_struc = triads["030T"][struc].nodes()
    neurons_in_030T.append(list(nodes_in_struc))

nr_min_one_030T = len(np.unique(neurons_in_030T))

neurons_in_111U = []
for struc in np.arange(len(triads["111U"])):
    nodes_in_struc = triads["111U"][struc].nodes()
    neurons_in_111U.append(list(nodes_in_struc))

nr_min_one_111U = len(np.unique(neurons_in_111U))

neurons_in_111D = []
for struc in np.arange(len(triads["111D"])):
    nodes_in_struc = triads["111D"][struc].nodes()
    neurons_in_111D.append(list(nodes_in_struc))

nr_min_one_111D = len(np.unique(neurons_in_111D))

neurons_in_120U = []
for struc in np.arange(len(triads["120U"])):
    nodes_in_struc = triads["120U"][struc].nodes()
    neurons_in_120U.append(list(nodes_in_struc))

nr_min_one_120U = len(np.unique(neurons_in_120U))

neurons_in_120D = []
for struc in np.arange(len(triads["120D"])):
    nodes_in_struc = triads["120D"][struc].nodes()
    neurons_in_120D.append(list(nodes_in_struc))

nr_min_one_120D = len(np.unique(neurons_in_120D))

neurons_in_120C = []
for struc in np.arange(len(triads["120C"])):
    nodes_in_struc = triads["120C"][struc].nodes()
    neurons_in_120C.append(list(nodes_in_struc))

nr_min_one_120C = len(np.unique(neurons_in_120C))

neurons_in_201 = []
for struc in np.arange(len(triads["201"])):
    nodes_in_struc = triads["201"][struc].nodes()
    neurons_in_201.append(list(nodes_in_struc))

nr_min_one_201 = len(np.unique(neurons_in_201))

neurons_in_210 = []
for struc in np.arange(len(triads["210"])):
    nodes_in_struc = triads["210"][struc].nodes()
    neurons_in_210.append(list(nodes_in_struc))

nr_min_one_210 = len(np.unique(neurons_in_210))

neurons_in_300 = []
for struc in np.arange(len(triads["300"])):
    nodes_in_struc = triads["300"][struc].nodes()
    neurons_in_300.append(list(nodes_in_struc))

nr_min_one_300 = len(np.unique(neurons_in_300))

neurons_in_030C = []
for struc in np.arange(len(triads["030C"])):
    nodes_in_struc = triads["030C"][struc].nodes()
    neurons_in_030C.append(list(nodes_in_struc))
nr_min_one_030C = len(np.unique(neurons_in_030C))


#######ACONN


triads = nx.triads_by_type(G_aconn)

neurons_in_021D = []
for struc in np.arange(len(triads["021D"])):
    nodes_in_struc = triads["021D"][struc].nodes()
    neurons_in_021D.append(list(nodes_in_struc))

nr_min_one_021D = len(np.unique(neurons_in_021D))

neurons_in_021U = []
for struc in np.arange(len(triads["021U"])):
    nodes_in_struc = triads["021U"][struc].nodes()
    neurons_in_021U.append(list(nodes_in_struc))

nr_min_one_021U = len(np.unique(neurons_in_021U))

neurons_in_021C = []
for struc in np.arange(len(triads["021C"])):
    nodes_in_struc = triads["021C"][struc].nodes()
    neurons_in_021C.append(list(nodes_in_struc))

nr_min_one_021C = len(np.unique(neurons_in_021C))

neurons_in_030T = []
for struc in np.arange(len(triads["030T"])):
    nodes_in_struc = triads["030T"][struc].nodes()
    neurons_in_030T.append(list(nodes_in_struc))

nr_min_one_030T = len(np.unique(neurons_in_030T))

neurons_in_111U = []
for struc in np.arange(len(triads["111U"])):
    nodes_in_struc = triads["111U"][struc].nodes()
    neurons_in_111U.append(list(nodes_in_struc))

nr_min_one_111U = len(np.unique(neurons_in_111U))

neurons_in_111D = []
for struc in np.arange(len(triads["111D"])):
    nodes_in_struc = triads["111D"][struc].nodes()
    neurons_in_111D.append(list(nodes_in_struc))

nr_min_one_111D = len(np.unique(neurons_in_111D))

neurons_in_120U = []
for struc in np.arange(len(triads["120U"])):
    nodes_in_struc = triads["120U"][struc].nodes()
    neurons_in_120U.append(list(nodes_in_struc))

nr_min_one_120U = len(np.unique(neurons_in_120U))

neurons_in_120D = []
for struc in np.arange(len(triads["120D"])):
    nodes_in_struc = triads["120D"][struc].nodes()
    neurons_in_120D.append(list(nodes_in_struc))

nr_min_one_120D = len(np.unique(neurons_in_120D))

neurons_in_120C = []
for struc in np.arange(len(triads["120C"])):
    nodes_in_struc = triads["120C"][struc].nodes()
    neurons_in_120C.append(list(nodes_in_struc))

nr_min_one_120C = len(np.unique(neurons_in_120C))

neurons_in_201 = []
for struc in np.arange(len(triads["201"])):
    nodes_in_struc = triads["201"][struc].nodes()
    neurons_in_201.append(list(nodes_in_struc))

nr_min_one_201 = len(np.unique(neurons_in_201))

neurons_in_210 = []
for struc in np.arange(len(triads["210"])):
    nodes_in_struc = triads["210"][struc].nodes()
    neurons_in_210.append(list(nodes_in_struc))

nr_min_one_210 = len(np.unique(neurons_in_210))

neurons_in_300 = []
for struc in np.arange(len(triads["300"])):
    nodes_in_struc = triads["300"][struc].nodes()
    neurons_in_300.append(list(nodes_in_struc))

nr_min_one_300 = len(np.unique(neurons_in_300))

neurons_in_030C = []
for struc in np.arange(len(triads["030C"])):
    nodes_in_struc = triads["030C"][struc].nodes()
    neurons_in_030C.append(list(nodes_in_struc))
nr_min_one_030C = len(np.unique(neurons_in_030C))


print("done")