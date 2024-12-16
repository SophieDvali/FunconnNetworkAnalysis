import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pickle
import networkx as nx



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
#levels_aconn = state_aconn.get_levels()

groups_heirarchical = []
for i in np.arange(len(neuronlist)):
    groups_heirarchical.append(levels[0].get_blocks()[i])

group_lengths = []
for gr in np.unique(groups_heirarchical):
    group_lengths.append(len(np.where(groups_heirarchical == gr)[0]))

group_sorter = np.argsort(group_lengths)

groups_heirarchical_unique = np.unique(groups_heirarchical)[group_sorter]



vncmotor = ["DA1", "DB1", "DB2", "AS1", "DD1", "VA1", "VB1", "VB2", "VD1", "VD2"]

locomotoryinterneurons = ["AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER"]



Amphid = ["ADFL", "ADLL", "AFDL", "ASEL", "ASGL", "ASHL", "ASIL", "ASJL", "ASKL", "AWAL", "AWBL", "AWCL", "ADFR",
          "ADLR", "AFDR","ASER", "ASGR", "ASHR", "ASIR", "ASJR", "ASKR", "AWAR", "AWBR", "AWCR"]



pharynx = ["I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6", "M1", "M2L", "M2R",
           "M3L", "M3R", "M4", "M5", "MCL", "MCR", "MI", "NSML", "NSMR"]

ringinterneurons = ["ADAL", "ADAR", "AIML", "AIMR", "AINL", "AINR", "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR",
                    "RID", "RIFL", "RIFR", "RIGL", "RIGR", "RIH", "RIPL", "RIPR", "RIR", "RIS", "RIVL", "RIVR",
                    "SAADL", "SAADR", "SAAVL", "SAAVR", "SABD", "SABVL", "SABVR"]


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

overlap_mechanosensor = np.zeros((len(groups_heirarchical_unique),1))
overlap_oxygensensors = np.zeros((len(groups_heirarchical_unique),1))
overlap_locomotion= np.zeros((len(groups_heirarchical_unique),1))
overlap_amphid = np.zeros((len(groups_heirarchical_unique),1))
overlap_vncmotor = np.zeros((len(groups_heirarchical_unique),1))
overlap_locomotoryinterneurons = np.zeros((len(groups_heirarchical_unique),1))
overlap_pharynx = np.zeros((len(groups_heirarchical_unique),1))
overlap_amphidinter = np.zeros((len(groups_heirarchical_unique),1))
overlap_ringinterneurons = np.zeros((len(groups_heirarchical_unique),1))
overlap_allother = np.zeros((len(groups_heirarchical_unique),1))

com6_amphid = []
com6_locomotoryinter = []
com6_amphid_ind = []
com6_locomotoryinter_ind = []
com6_sensory_ind = []
com6_inter_ind = []
com6_motor_ind = []
com4_VNC_ind = []
com4_ring_ind = []
for i in np.arange(len(groups_heirarchical_unique)):
    indicies_of_group = np.where(groups_heirarchical == groups_heirarchical_unique[i])[0]
    neurons_in_group = np.array(neuronlist)[indicies_of_group]
    print("Group:"+str(i+1))
    print(neurons_in_group)
    for n in np.arange(len(neurons_in_group)):
        neuron = neurons_in_group[n]
        if neuron in Amphid:
            overlap_amphid[i] += 1
            if i == 5:
                com6_amphid.append(neuron)
                com6_amphid_ind.append(indicies_of_group[n])
        if neuron in vncmotor:
            overlap_vncmotor[i] += 1
            if i == 3:
                com4_VNC_ind.append(indicies_of_group[n])
        if neuron in locomotoryinterneurons:
            overlap_locomotoryinterneurons[i] += 1
            if i == 5:
                com6_locomotoryinter.append(neuron)
                com6_locomotoryinter_ind.append(indicies_of_group[n])
        if neuron in pharynx:
            overlap_pharynx[i] += 1
        if neuron in ringinterneurons:
            overlap_ringinterneurons[i] += 1
            if i == 3:
                com4_ring_ind.append(indicies_of_group[n])



##### in community 6 we have amphid sensory, locomotory inter and other
#0: amphid sensory, 1: locomotory inter, 2: other
com6_other = list(set(np.where(groups_heirarchical == groups_heirarchical_unique[5])[0]) - set(com6_amphid_ind) - set(com6_locomotoryinter_ind))
number_connections = np.zeros((3,3))

neurontypes = [np.array(com6_amphid_ind), np.array(com6_locomotoryinter_ind), np.array(com6_other)]

number_connections_norm = np.zeros((3, 3))

for i in np.arange(3):
    for j in np.arange(3):

        groupi = neurontypes[i]
        groupj = neurontypes[j]

        for neu_i in groupi:
            for neu_j in groupj:
                if g.edge(neu_j,neu_i):
                    number_connections[i,j] += 1

        number_connections_norm[i,j] = number_connections[i,j]/(len(groupi)*len(groupj))


G = nx.DiGraph(get_adj_matrix)


com_6_inds = np.where(groups_heirarchical == groups_heirarchical_unique[5])[0]
com1_inds = np.where(groups_heirarchical == groups_heirarchical_unique[0])[0]

nr_6_to_6_connections = 0
nr_involving_amphid = 0
nr_involving_locomotory = 0
nr_involving_sensory = 0
nr_involving_locomotory_or_amphid = 0
nr_sensory_to_inter = 0
nr_inter_to_inter = 0
nr_into_pharynx = 0
nr_outof_pharynx = 0
nr_outof_amphid = 0
nr_into_amphid = 0

all_edges = [e for e in G.edges]

for edg in all_edges:
    source = edg[0]
    target = edg[1]
    source_neuron = neuronlist[source]
    target_neuron = neuronlist[target]

    if source in com_6_inds and target in com_6_inds:
        nr_6_to_6_connections += 1

        if source in com6_amphid_ind or target in com6_amphid_ind:
            nr_involving_amphid += 1

        if source in com6_locomotoryinter_ind or target in com6_locomotoryinter_ind:
            nr_involving_locomotory += 1

        if source_neuron in sensory or target_neuron in sensory:
            nr_involving_sensory += 1

        if source in com6_locomotoryinter_ind or target in com6_locomotoryinter_ind or source in com6_amphid_ind or target in com6_amphid_ind:
            nr_involving_locomotory_or_amphid += 1

        if source_neuron in sensory and target_neuron in inter:
            nr_sensory_to_inter += 1
        if source_neuron in inter and target_neuron in inter:
            nr_inter_to_inter += 1

    if source_neuron in pharynx and target_neuron not in pharynx:
        nr_outof_pharynx += 1

    if source_neuron not in pharynx and target_neuron in pharynx:
        nr_into_pharynx += 1

    if source_neuron in Amphid and target_neuron not in Amphid:
        nr_outof_amphid += 1

    if source_neuron not in Amphid and target_neuron in Amphid:
        nr_into_amphid += 1











plt.figure()
plt.imshow(number_connections_norm, cmap = "Blues")
plt.xlabel("From")
plt.ylabel("To")
plt.xticks(np.arange(3), ["Amphid sensory", "Locomotory inter",
                          "Other"])
plt.yticks(np.arange(3), ["Amphid sensory", "Locomotory inter",
                          "Other"])
plt.xticks(rotation=90)
plt.colorbar()
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/enrichment_edge_check_6.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig2/enrichment_edge_check_6.eps",
    dpi=300, bbox_inches="tight", format ="eps")
plt.show()

plt.figure()
plt.imshow(number_connections, cmap = "Blues")
plt.xlabel("From")
plt.ylabel("To")
plt.xticks(np.arange(3), ["Amphid sensory", "Locomotory inter",
                          "Other"])
plt.yticks(np.arange(3), ["Amphid sensory", "Locomotory inter",
                          "Other"])
plt.xticks(rotation=90)
plt.colorbar()
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/enrichment_edge_check_6_nr.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig2/enrichment_edge_check_6_nr.eps",
    dpi=300, bbox_inches="tight", format ="eps")
plt.show()

### in community 4 we have ring interneurons and vnc motor neurons
## 0: ring inter, 1: vnc motor, 2: other
com4_other = list(set(np.where(groups_heirarchical == groups_heirarchical_unique[3])[0]) - set(com4_VNC_ind) - set(com4_ring_ind))
number_connections_4 = np.zeros((3,3))

neurontypes_4 = [np.array(com4_ring_ind), np.array(com4_VNC_ind), np.array(com4_other)]

number_connections_norm_4 = np.zeros((3, 3))



for i in np.arange(3):
    for j in np.arange(3):

        groupi = neurontypes_4[i]
        groupj = neurontypes_4[j]

        for neu_i in groupi:
            for neu_j in groupj:
                if g.edge(neu_j,neu_i):
                    number_connections_4[i,j] += 1

        number_connections_norm_4[i,j] = number_connections_4[i,j]/(len(groupi)*len(groupj))

plt.figure()
plt.imshow(number_connections_norm_4, cmap = "Blues")
plt.xlabel("From")
plt.ylabel("To")
plt.xticks(np.arange(3), ["Ring inter", "VNC motor",
                          "Other"])
plt.yticks(np.arange(3), ["Ring inter", "VNC motor",
                          "Other"])
plt.xticks(rotation=90)
plt.colorbar()
plt.tight_layout()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/enrichment_edge_check_4.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
    "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig2/enrichment_edge_check_4.eps",
    dpi=300, bbox_inches="tight", format ="eps")
plt.show()

print("done")
