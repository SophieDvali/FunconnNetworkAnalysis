import graph_tool.all as gt
import pickle



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


state = gt.minimize_nested_blockmodel_dl(g)

state_aconn = gt.minimize_nested_blockmodel_dl(g_aconn)

# Now we run 1000 sweeps of the MCMC

dS, nmoves = 0, 0
for i in range(100):
    ret = state.multiflip_mcmc_sweep(niter=10)
    dS += ret[0]
    nmoves += ret[1]

print("Change in description length:", dS)
print("Number of accepted vertex moves:", nmoves)

dS_aconn, nmoves_aconn = 0, 0
for i in range(100):
    ret = state_aconn.multiflip_mcmc_sweep(niter=10)
    dS_aconn += ret[0]
    nmoves_aconn += ret[1]

print("Change in description length:", dS_aconn)
print("Number of accepted vertex moves:", nmoves_aconn)


#for the funconn
# We will first equilibrate the Markov chain
gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10))

# collect nested partitions
bs = []

def collect_partitions(s):
   global bs
   bs.append(s.get_bs())

# Now we collect the marginals for exactly 100,000 sweeps
gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                    callback=collect_partitions)

# Disambiguate partitions and obtain marginals
pmode = gt.PartitionModeState(bs, nested=True, converge=True)
pv = pmode.get_marginal(g)

# Get consensus estimate
bs = pmode.get_max_nested()

state = state.copy(bs=bs)

# We can visualize the marginals as pie charts on the nodes:
state.draw(vertex_shape="pie", vertex_pie_fractions=pv,
           output="funcon-nested-sbm-marginals_nonames.pdf")
state.draw(vertex_text =g.vp.neuron, vertex_shape="pie", vertex_pie_fractions=pv, vertex_text_position='centered',
           output="funcon-nested-sbm-marginals.pdf")
state.draw(vertex_text =g.vp.neuron, vertex_text_position='centered',
           output="funcon-nested-sbm-marginals_nopie.pdf")

#for the aconn

# We will first equilibrate the Markov chain
gt.mcmc_equilibrate(state_aconn, wait=1000, mcmc_args=dict(niter=10))

# collect nested partitions
bs_aconn = []

def collect_partitions_aconn(s):
   global bs_aconn
   bs_aconn.append(s.get_bs())

# Now we collect the marginals for exactly 100,000 sweeps
gt.mcmc_equilibrate(state_aconn, force_niter=10000, mcmc_args=dict(niter=10),
                    callback=collect_partitions_aconn)

# Disambiguate partitions and obtain marginals
pmode_aconn = gt.PartitionModeState(bs_aconn, nested=True, converge=True)
pv_aconn = pmode_aconn.get_marginal(g_aconn)

# Get consensus estimate
bs_aconn = pmode_aconn.get_max_nested()

state_aconn = state_aconn.copy(bs=bs_aconn)

# We can visualize the marginals as pie charts on the nodes:
state_aconn.draw(vertex_shape="pie", vertex_pie_fractions=pv_aconn,
           output="aconn-nested-sbm-marginals_nonames.pdf")
state_aconn.draw(vertex_text =g_aconn.vp.neuron, vertex_shape="pie", vertex_text_position='centered', vertex_pie_fractions=pv_aconn,
           output="aconn-nested-sbm-marginals.pdf")
state_aconn.draw(vertex_text =g_aconn.vp.neuron, vertex_text_position='centered',
           output="aconn-nested-sbm-marginals_nopie.pdf")

with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/fconn_state_mcmc.pickle', 'wb') as st:
    pickle.dump(state, st, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/sophiedvali/Documents/University/Research/NetworkAnalysis/SMBthings/aconn_state_mcmc.pickle', 'wb') as st_aconn:
    pickle.dump(state_aconn, st_aconn, protocol=pickle.HIGHEST_PROTOCOL)

print("done")