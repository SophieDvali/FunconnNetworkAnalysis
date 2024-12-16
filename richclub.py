import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys
from matplotlib_venn import venn3

calculate_phi_rand = "--calculate-phi-rand" in sys.argv


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

#### FOR FCONN####

degree = []
indegree =[]
outdegree = []
for v in g.vertices():
    degree.append(v.out_degree() + v.in_degree())
    indegree.append(v.in_degree())
    outdegree.append(v.out_degree())

kmax = np.max(degree)
kmin = np.min(degree)
phi = np.empty((1,kmax))
phi.fill(np.nan)


for k in np.arange(kmax):
    nodes_higher_than_k_indicies = np.where(degree > k)
    number_nodes_higher_than_k = len(nodes_higher_than_k_indicies[0])
    if number_nodes_higher_than_k > 1:
        Ek = 0
        for i in np.arange(number_nodes_higher_than_k):
            for j in np.arange(number_nodes_higher_than_k):
                if g.edge(nodes_higher_than_k_indicies[0][j],nodes_higher_than_k_indicies[0][i]) != None:
                    #egdes = q_head_binary[nodes_higher_than_k_indicies[0][i],nodes_higher_than_k_indicies[0][j]]
                    Ek = Ek + 1

        Nk = number_nodes_higher_than_k
        phi[0][k] = (Ek)/(Nk*(Nk-1))


kmax_in = np.max(indegree)
kmin_in = np.min(indegree)
phi_in = np.empty((1,kmax_in))
phi_in.fill(np.nan)

for k in np.arange(kmax_in):
    nodes_higher_than_k_indicies = np.where(indegree > k)
    number_nodes_higher_than_k = len(nodes_higher_than_k_indicies[0])
    if number_nodes_higher_than_k > 1:
        Ek = 0
        for i in np.arange(number_nodes_higher_than_k):
            for j in np.arange(number_nodes_higher_than_k):
                if g.edge(nodes_higher_than_k_indicies[0][j],nodes_higher_than_k_indicies[0][i]) != None:
                    #egdes = q_head_binary[nodes_higher_than_k_indicies[0][i],nodes_higher_than_k_indicies[0][j]]
                    Ek = Ek + 1

        Nk = number_nodes_higher_than_k
        phi_in[0][k] = (Ek)/(Nk*(Nk-1))


kmax_out = np.max(outdegree)
kmin_out = np.min(outdegree)
phi_out = np.empty((1,kmax_out))
phi_out.fill(np.nan)

for k in np.arange(kmax_out):
    nodes_higher_than_k_indicies = np.where(indegree > k)
    number_nodes_higher_than_k = len(nodes_higher_than_k_indicies[0])
    if number_nodes_higher_than_k > 1:
        Ek = 0
        for i in np.arange(number_nodes_higher_than_k):
            for j in np.arange(number_nodes_higher_than_k):
                if g.edge(nodes_higher_than_k_indicies[0][j],nodes_higher_than_k_indicies[0][i]) != None:
                    #egdes = q_head_binary[nodes_higher_than_k_indicies[0][i],nodes_higher_than_k_indicies[0][j]]
                    Ek = Ek + 1

        Nk = number_nodes_higher_than_k
        phi_out[0][k] = (Ek)/(Nk*(Nk-1))


g_rand = gt.Graph(g)

#make randomized networks method configuration

if calculate_phi_rand:
    phi_rand = np.empty((500,kmax))
    phi_rand.fill(np.nan)
    phi_rand_in = np.empty((500, kmax_in))
    phi_rand_in.fill(np.nan)
    phi_rand_out = np.empty((500, kmax_out))
    phi_rand_out.fill(np.nan)

    for iter in np.arange(500):
        gt.random_rewire(g_rand, "configuration")
        degree_rand = []
        indegree_rand = []
        outdegree_rand = []
        for v in g_rand.vertices():
            degree_rand.append(v.out_degree() + v.in_degree())
            indegree_rand.append(v.in_degree())
            outdegree_rand.append(v.out_degree())

        kmax = np.max(degree_rand)
        kmin = np.min(degree_rand)

        kmax_in = np.max(indegree_rand)
        kmax_out = np.max(outdegree_rand)

        for k in np.arange(kmax):
            nodes_higher_than_k_indicies = np.where(degree_rand > k)
            number_nodes_higher_than_k = len(nodes_higher_than_k_indicies[0])
            if number_nodes_higher_than_k > 1:
                Ek = 0
                for i in np.arange(number_nodes_higher_than_k):
                    for j in np.arange(number_nodes_higher_than_k):
                        if g_rand.edge(nodes_higher_than_k_indicies[0][j], nodes_higher_than_k_indicies[0][i]) != None:
                            # egdes = q_head_binary[nodes_higher_than_k_indicies[0][i],nodes_higher_than_k_indicies[0][j]]
                            Ek = Ek + 1

                Nk = number_nodes_higher_than_k
                phi_rand[iter][k] = (Ek) / (Nk * (Nk - 1))

        for k in np.arange(kmax_in):
            nodes_higher_than_k_indicies = np.where(indegree_rand > k)
            number_nodes_higher_than_k = len(nodes_higher_than_k_indicies[0])
            if number_nodes_higher_than_k > 1:
                Ek = 0
                for i in np.arange(number_nodes_higher_than_k):
                    for j in np.arange(number_nodes_higher_than_k):
                        if g.edge(nodes_higher_than_k_indicies[0][j], nodes_higher_than_k_indicies[0][i]) != None:
                            # egdes = q_head_binary[nodes_higher_than_k_indicies[0][i],nodes_higher_than_k_indicies[0][j]]
                            Ek = Ek + 1

                Nk = number_nodes_higher_than_k
                phi_rand_in[iter][k] = (Ek) / (Nk * (Nk - 1))

        for k in np.arange(kmax_out):
            nodes_higher_than_k_indicies = np.where(outdegree_rand > k)
            number_nodes_higher_than_k = len(nodes_higher_than_k_indicies[0])
            if number_nodes_higher_than_k > 1:
                Ek = 0
                for i in np.arange(number_nodes_higher_than_k):
                    for j in np.arange(number_nodes_higher_than_k):
                        if g.edge(nodes_higher_than_k_indicies[0][j], nodes_higher_than_k_indicies[0][i]) != None:
                            # egdes = q_head_binary[nodes_higher_than_k_indicies[0][i],nodes_higher_than_k_indicies[0][j]]
                            Ek = Ek + 1

                Nk = number_nodes_higher_than_k
                phi_rand_out[iter][k] = (Ek) / (Nk * (Nk - 1))

    df_phirand = pd.DataFrame(phi_rand)
    df_phirand.to_csv('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/phi_rand_config.csv')

else:
    df_phirand = pd.read_csv(
        '/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/phi_rand_config.csv')
    phi_rand = df_phirand.to_numpy()
    phi_rand = np.delete(phi_rand, obj=0, axis=1)


mean_phi_rand = np.mean(phi_rand, axis = 0)
phi_norm = phi[0]/mean_phi_rand
std_phi_rand = np.std(phi_rand, axis = 0)


# greater_than_five_sigma = phi_norm >= 1 + 5*std_phi_rand
# greater_than_one_sigma = phi_norm >= 1 + std_phi_rand
# greater_than_ten_sigma = phi_norm >= 1 + 10*std_phi_rand


fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.arange(len(phi[0])), phi[0], color = 'cornflowerblue', marker = ".", linestyle = "-", label= "$\Phi$")
plt.errorbar(np.arange(len(phi[0])), mean_phi_rand, yerr = std_phi_rand, color = 'slateblue', marker = ".", linestyle = "-", label= "$<\Phi_{rand}>$")
plt.plot(np.arange(len(phi[0])), phi_norm, color = 'darkorchid', marker = ".", linestyle = "-", label= "$\Phi_{norm}$")
plt.legend()
plt.axhline(1, color= "black")
plt.ylabel("$\Phi (k)$")
plt.xlabel("k")
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/richclub_plot_config.pdf", dpi=300, bbox_inches="tight")
plt.show()
plt.clf()

#look for local maxima in the Phi norm to find two rich clubs:
#bye eye: k = 3 k = 25, and k = 40

neurons_k_gr_3 = neuronlist[np.where(np.array(degree) > 3)[0]]
indicies_k_gr_3 = np.where(np.array(degree) > 3)[0]

nr_k_gr_3 = len(indicies_k_gr_3)
percent_k_gr_3 = nr_k_gr_3/188


neurons_k_gr_25 = neuronlist[np.where(np.array(degree) > 25)[0]]
indicies_k_gr_25 = np.where(np.array(degree) > 25)[0]

nr_k_gr_25 = len(indicies_k_gr_25)
percent_k_gr_25 = nr_k_gr_25/188


neurons_k_gr_40 = neuronlist[np.where(np.array(degree) > 40)[0]]
indicies_k_gr_40 = np.where(np.array(degree) > 40)[0]

nr_k_gr_40 = len(indicies_k_gr_40)
percent_k_gr_40 = nr_k_gr_40/188


#check for the rich club enrichment for SIM

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

#calculating the overlap between motor sensory and inter with the rich clubs
#1 is sensory, 2 is inter, 3 is motor
overlap_SIM = np.zeros((3,3))
overlap_pharynx = np.zeros((3,1))

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

for i in np.arange(3):
    for j in np.arange(3):
        if i == 0:
            SIM = sensory
        elif i == 1:
            SIM = inter
        elif i == 2:
            SIM = motor

        if j == 0:
            d = neurons_k_gr_3
        elif j == 1:
            d = neurons_k_gr_25
        elif j == 2:
            d = neurons_k_gr_40
        overlap_SIM[i, j] = len(intersection(SIM, d))
        overlap_pharynx[j,0] =  len(intersection(pharynx, d))




percentage_SIM =  np.zeros((3,3))
percentage_pharynx = np.zeros((3,1))

for j in np.arange(3):
    percentage_SIM[:, j] = overlap_SIM[:, j] / np.sum(overlap_SIM, axis=0)[j]
    percentage_pharynx[j] = overlap_pharynx[j]/ np.sum(overlap_SIM, axis=1)[j]


fig, ax = plt.subplots()
ax.imshow(overlap_SIM, cmap=plt.cm.Blues)
for i in np.arange(3):
    for j in np.arange(3):
        c = overlap_SIM[i,j]
        ax.text(j, i, str(c), va='center', ha='center')
ax.set_yticklabels(["", "Sensoryneuron", "","Interneuron", "","Motorneuron"])
ax.set_xticklabels(["", "k>3","", "k>25","", "k>40"])
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment_richclub.pdf", dpi=300, bbox_inches="tight")
plt.show()



fig, ax = plt.subplots()
ax.imshow(percentage_SIM[:,1:], cmap=plt.cm.Blues, vmin = 0, vmax = 1)
for i in np.arange(3):
    for j in [0,1]:
        c = percentage_SIM[i,j]
        ax.text(j, i, str(round(c * 100, 1)) + "%", va='center', ha='center', color = "black")
ax.set_yticklabels(["", "Sensoryneuron", "","Interneuron", "","Motorneuron"])
ax.set_xticklabels(["", "k>25","", "k>40"])
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment_richclub_percent.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig5/SIM_enrichment_richclub_percent.eps",
        dpi=300, bbox_inches="tight", format="eps")
plt.show()



#### FOR ACONN####


degree_aconn = []
for v in g_aconn.vertices():
    degree_aconn.append(v.out_degree() + v.in_degree())

kmax_aconn = np.max(degree_aconn)
kmin_aconn = np.min(degree_aconn)
phi_aconn = np.empty((1,kmax_aconn))
phi_aconn.fill(np.nan)

for k in np.arange(kmax_aconn):
    nodes_higher_than_k_indicies_aconn = np.where(degree_aconn > k)
    number_nodes_higher_than_k_aconn = len(nodes_higher_than_k_indicies_aconn[0])
    if number_nodes_higher_than_k_aconn > 1:
        Ek = 0
        for i in np.arange(number_nodes_higher_than_k_aconn):
            for j in np.arange(number_nodes_higher_than_k_aconn):
                if g_aconn.edge(nodes_higher_than_k_indicies_aconn[0][j],nodes_higher_than_k_indicies_aconn[0][i]) != None:
                    #egdes = q_head_binary[nodes_higher_than_k_indicies[0][i],nodes_higher_than_k_indicies[0][j]]
                    Ek = Ek + 1

        Nk = number_nodes_higher_than_k_aconn
        phi_aconn[0][k] = (Ek)/(Nk*(Nk-1))

g_rand_aconn = gt.Graph(g_aconn)

#make randomized networks method configuration

if True:
    phi_rand_aconn = np.empty((100,kmax_aconn))
    phi_rand_aconn.fill(np.nan)

    for iter in np.arange(100):
        gt.random_rewire(g_rand_aconn, "configuration")
        degree_rand_aconn = []
        for v in g_rand_aconn.vertices():
            degree_rand_aconn.append(v.out_degree() + v.in_degree())

        kmax_aconn = np.max(degree_rand_aconn)
        kmin_aconn = np.min(degree_rand_aconn)

        for k in np.arange(kmax_aconn):
            nodes_higher_than_k_indicies_aconn = np.where(degree_rand_aconn > k)
            number_nodes_higher_than_k_aconn = len(nodes_higher_than_k_indicies_aconn[0])
            if number_nodes_higher_than_k_aconn > 1:
                Ek = 0
                for i in np.arange(number_nodes_higher_than_k_aconn):
                    for j in np.arange(number_nodes_higher_than_k_aconn):
                        if g_rand_aconn.edge(nodes_higher_than_k_indicies_aconn[0][j], nodes_higher_than_k_indicies_aconn[0][i]) != None:
                            # egdes = q_head_binary[nodes_higher_than_k_indicies[0][i],nodes_higher_than_k_indicies[0][j]]
                            Ek = Ek + 1

                Nk = number_nodes_higher_than_k_aconn
                phi_rand_aconn[iter][k] = (Ek) / (Nk * (Nk - 1))
    df_phirand_aconn = pd.DataFrame(phi_rand_aconn)
    df_phirand_aconn.to_csv('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/phi_rand_config_aconn.csv')



mean_phi_rand_aconn = np.mean(phi_rand_aconn, axis = 0)
phi_norm_aconn = phi_aconn[0]/mean_phi_rand_aconn
std_phi_rand_aconn = np.std(phi_rand_aconn, axis = 0)

greater_than_five_sigma_aconn = phi_norm_aconn >= 1 + 5*std_phi_rand_aconn
greater_than_one_sigma_aconn = phi_norm_aconn >= 1 + std_phi_rand_aconn
greater_than_ten_sigma_aconn = phi_norm_aconn >= 1 + 10*std_phi_rand_aconn


fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.arange(len(phi_aconn[0])), phi_aconn[0], color = 'cornflowerblue', marker = ".", linestyle = "-", label= "$\Phi$")
plt.errorbar(np.arange(len(phi_aconn[0])), mean_phi_rand_aconn, yerr = std_phi_rand_aconn, color = 'slateblue', marker = ".", linestyle = "-", label= "$<\Phi_{rand}>$")
plt.plot(np.arange(len(phi_aconn[0])), phi_norm_aconn, color = 'darkorchid', marker = ".", linestyle = "-", label= "$\Phi_{norm}$")
plt.legend()
plt.axhline(1, color= "black")
ax.fill_betweenx([0,2.4], 62, 93, color='salmon', alpha=0.3)
plt.ylabel("$\Phi (k)$")
plt.xlabel("k")
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/richclub_plot_config_aconn.pdf", dpi=300, bbox_inches="tight")
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig5/richclub_plot_config_aconn.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.clf()


neurons_k_gr_62_aconn = neuronlist[np.where(np.array(degree_aconn) > 62)[0]]
indicies_k_gr_62_aconn = np.where(np.array(degree_aconn) > 62)[0]

# neurons_k_gr_71_aconn = neuronlist[np.where(np.array(degree_aconn) > 71)[0]]
# indicies_k_gr_71_aconn = np.where(np.array(degree_aconn) > 71)[0]

set1 = set(neurons_k_gr_62_aconn)
set2 = set(neurons_k_gr_25)
set3 = set(neurons_k_gr_40)
venn3([set1, set2, set3], ('Aconn', 'Fconn 25', 'Fconn 40'))
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/richclub_plot_venn_digram.pdf", dpi=300, bbox_inches="tight")

plt.show()


overlap_SIM_aconn = np.zeros((3,1))
for i in np.arange(3):
    if i == 0:
        SIM = sensory
    elif i == 1:
        SIM = inter
    elif i == 2:
        SIM = motor
    overlap_SIM_aconn[i] = len(intersection(SIM, neurons_k_gr_62_aconn))



percentage_SIM_aconn =  np.zeros((3,1))



percentage_SIM_aconn = overlap_SIM_aconn / np.sum(overlap_SIM_aconn, axis=0)[0]



fig, ax = plt.subplots()
ax.imshow(percentage_SIM_aconn, cmap=plt.cm.Blues, vmin = 0, vmax = 1)
for i in np.arange(3):
    c = percentage_SIM_aconn[i][0]
    if i == 1:
        ax.text(0, i, str(round(c * 100, 1)) + "%", va='center', ha='center', color = "white")
    else:
        ax.text(0, i, str(round(c * 100, 1)) + "%", va='center', ha='center', color = "black")

ax.set_yticklabels(["", "Sensoryneuron", "","Interneuron", "","Motorneuron"])
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/SIM_enrichment_richclub_percent_aconn.pdf", dpi=300, bbox_inches="tight")
plt.savefig(
        "/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/ForFigures/Fig5/SIM_enrichment_richclub_percent_aconn.eps",
        dpi=300, bbox_inches="tight", format="eps")
plt.show()


print("Done")