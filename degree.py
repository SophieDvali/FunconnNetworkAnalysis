import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy as sp
from scipy.stats import norm
import sys
import matplotlib


#look at the distributions

matplotlib.rcParams['pdf.fonttype']=42

calculate_phi_rand = "--calculate-phi-rand" in sys.argv

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


sorter_indegree = np.argsort(indegree)
sorter_outdegree = np.argsort(outdegree)

above_20_indegree = neuronlist[np.where(np.array(indegree) > 20)[0]]
above_20_outdegree = neuronlist[np.where(np.array(outdegree) > 20)[0]]

degree = np.array(degree)
degree_aconn = np.array(degree_aconn)

slope1, intercept1, r_value1, p_value1, std_err1 = sp.stats.linregress(degree_aconn, degree)

plt.figure()
plt.scatter(degree_aconn,degree, color = "black")
#plt.scatter(degree_aconn[np.where(np.array(degree_aconn) > 62)[0]], degree[np.where(np.array(degree_aconn) > 62)[0]], color = "salmon")
#plt.scatter(degree_aconn[np.where(np.array(degree) > 25)[0]], degree[np.where(np.array(degree) > 25)[0]], color = "seagreen")
#plt.scatter(degree_aconn[np.where(np.array(degree) > 40)[0]], degree[np.where(np.array(degree) > 40)[0]], color = "cornflowerblue")
#plt.scatter(degree_aconn[[45,46]], degree[[45,46]], color = "blueviolet")
plt.fill_between(np.arange(0,63),25,40, color = "seagreen", alpha = 0.3)
plt.fill_between(np.arange(0,63),40,53, color = "cornflowerblue", alpha = 0.3)
plt.fill_between(np.arange(62,95),0,40, color = "salmon", alpha = 0.3)
plt.fill_between(np.arange(62,95),40,53, color = "blueviolet", alpha = 0.3)
plt.xlabel("Degree Anatomical", fontsize = 15)
plt.ylabel("Degree Signal Propagation", fontsize = 15)
plt.xticks([0,50,100], [0,50,100], fontsize = 15)
plt.yticks([0,25,50], [0,25,50], fontsize = 15)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/Aconndegree_vs_fconndegree.pdf", dpi=300, bbox_inches="tight")
plt.show()


for i in np.arange(len(degree)):
    if degree[i] > 40 and degree_aconn[i] > 70:
        print(neuronlist[i])
        print(degree[i])
        print(degree_aconn[i])
        print(i)

a, b = np.polyfit(observations, degree, 1)

slope, intercept, r_value, p_value, std_err = sp.stats.linregress(observations, degree)



errors = []
for i  in np.arange(len(degree)):
    y = degree[i]
    p_of_x = a*observations[i]+b
    errors.append(p_of_x - y)



(mu, sigma) = norm.fit(errors)
y_gauss = norm.pdf( np.arange(-25, 25, 1), mu, sigma)

y_line = a*observations+b
y_line_ps = a*np.arange(min(observations),max(observations),1)+b+(1*sigma)
y_line_ms = a*np.arange(min(observations),max(observations),1)+b-(1*sigma)

plt.figure()
plt.scatter(observations,degree, color = "black")
plt.scatter(observations[np.where(np.array(degree_aconn) > 62)[0]], degree[np.where(np.array(degree_aconn) > 62)[0]], color = "salmon")
plt.scatter(observations[np.where(np.array(degree) > 25)[0]], degree[np.where(np.array(degree) > 25)[0]], color = "seagreen")
plt.scatter(observations[np.where(np.array(degree) > 40)[0]], degree[np.where(np.array(degree) > 40)[0]], color = "cornflowerblue")
plt.scatter(observations[[45,46]], degree[[45,46]], color = "blueviolet")
plt.plot(observations, a*observations+b, color = "red")
plt.fill_between(np.arange(min(observations),max(observations),1), y_line_ms, y_line_ps, color = "grey", alpha = 0.3)
plt.xlabel("Number of Observations", fontsize = 15)
plt.ylabel("Degree Signal Propagation", fontsize = 15)
plt.xticks([0,50,90], [0,50,90], fontsize = 15)
plt.yticks([0,25,50], [0,25,50], fontsize = 15)
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/obs_vs_fconndegree.pdf", dpi=300, bbox_inches="tight")
plt.show()




plt.figure()
plt.hist(errors, density = True, alpha = 0.5, color = "grey")
l_gauss = plt.plot(np.arange(-25, 25, 1), y_gauss,color = "grey",linestyle =  '--', linewidth=2)
plt.hist(np.array(errors)[np.where(np.array(degree_aconn) > 62)[0]], density = True, alpha = 0.5, color = "salmon", label= "Anatomy")
plt.hist(np.array(errors)[np.where(np.array(degree) > 25)[0]], density = True, alpha = 0.5, color = "seagreen", label= "Signal Prop >25")
plt.hist(np.array(errors)[np.where(np.array(degree) > 40)[0]], density = True, alpha = 0.5, color = "cornflowerblue",  label= "Signal Prop >40")
plt.axvline(np.array(errors)[45], color = "blueviolet")
plt.axvline(np.array(errors)[46], color = "blueviolet")
plt.xlabel("Error",fontsize = 15)
plt.ylabel("Density",fontsize = 15)
plt.xticks([-20,0,20], [-20,0,20], fontsize = 15)
plt.yticks([0,0.1,0.2], [0.0,0.1,0.2], fontsize = 15)
plt.legend()
plt.savefig("/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/obs_vs_fconndegree_error_hist.pdf", dpi=300, bbox_inches="tight")
plt.show()



print("done")


