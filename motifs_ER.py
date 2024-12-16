
import networkx as nx
import numpy as np

##Calculating the three node motifs in random ER networks for anatomy and signaling


rand_neurons_in_021D = []
ran_nr_min_one_021D = []

rand_neurons_in_021D_aconn = []
ran_nr_min_one_021D_aconn = []

rand_neurons_in_021U = []
ran_nr_min_one_021U = []

rand_neurons_in_021U_aconn = []
ran_nr_min_one_021U_aconn = []

rand_neurons_in_021C = []
ran_nr_min_one_021C = []

rand_neurons_in_021C_aconn = []
ran_nr_min_one_021C_aconn = []

rand_neurons_in_030T = []
ran_nr_min_one_030T = []

rand_neurons_in_030T_aconn = []
ran_nr_min_one_030T_aconn = []

rand_neurons_in_111U = []
ran_nr_min_one_111U = []

rand_neurons_in_111U_aconn = []
ran_nr_min_one_111U_aconn = []

rand_neurons_in_111D = []
ran_nr_min_one_111D = []

rand_neurons_in_111D_aconn = []
ran_nr_min_one_111D_aconn = []

rand_neurons_in_120U = []
ran_nr_min_one_120U = []

rand_neurons_in_120U_aconn = []
ran_nr_min_one_120U_aconn = []

rand_neurons_in_120D = []
ran_nr_min_one_120D = []

rand_neurons_in_120D_aconn = []
ran_nr_min_one_120D_aconn = []

rand_neurons_in_120C = []
ran_nr_min_one_120C = []

rand_neurons_in_120C_aconn = []
ran_nr_min_one_120C_aconn = []

rand_neurons_in_201 = []
ran_nr_min_one_201 = []

rand_neurons_in_201_aconn = []
ran_nr_min_one_201_aconn = []

rand_neurons_in_210 = []
ran_nr_min_one_210 = []

rand_neurons_in_210_aconn = []
ran_nr_min_one_210_aconn = []

rand_neurons_in_300 = []
ran_nr_min_one_300 = []

rand_neurons_in_300_aconn = []
ran_nr_min_one_300_aconn = []


rand_neurons_in_030C = []
ran_nr_min_one_030C = []

rand_neurons_in_030C_aconn = []
ran_nr_min_one_030C_aconn = []

for i in np.arange(5):

    #random ER graphs with the same number of nodes and denisty as signaling and anatomical networks

    G_ER = nx.erdos_renyi_graph(188, 0.033, directed= True)
    G_aconn_ER = nx.erdos_renyi_graph(188, 0.092,  directed= True)

    triads = nx.triads_by_type(G_ER)

    neurons_in_021D = []
    for struc in np.arange(len(triads["021D"])):
        nodes_in_struc = triads["021D"][struc].nodes()
        neurons_in_021D.append(list(nodes_in_struc))

    nr_min_one_021D = len(np.unique(neurons_in_021D))
    rand_neurons_in_021D.append(len(triads["021D"]))
    ran_nr_min_one_021D.append(nr_min_one_021D)

    neurons_in_021U = []
    for struc in np.arange(len(triads["021U"])):
        nodes_in_struc = triads["021U"][struc].nodes()
        neurons_in_021U.append(list(nodes_in_struc))

    nr_min_one_021U = len(np.unique(neurons_in_021U))
    rand_neurons_in_021U.append(len(triads["021U"]))
    ran_nr_min_one_021U.append(nr_min_one_021U)

    neurons_in_021C = []
    for struc in np.arange(len(triads["021C"])):
        nodes_in_struc = triads["021C"][struc].nodes()
        neurons_in_021C.append(list(nodes_in_struc))

    nr_min_one_021C = len(np.unique(neurons_in_021C))
    rand_neurons_in_021C.append(len(triads["021C"]))
    ran_nr_min_one_021C.append(nr_min_one_021C)

    neurons_in_030T = []
    for struc in np.arange(len(triads["030T"])):
        nodes_in_struc = triads["030T"][struc].nodes()
        neurons_in_030T.append(list(nodes_in_struc))

    nr_min_one_030T = len(np.unique(neurons_in_030T))
    rand_neurons_in_030T.append(len(triads["030T"]))
    ran_nr_min_one_030T.append(nr_min_one_030T)

    neurons_in_111U = []
    for struc in np.arange(len(triads["111U"])):
        nodes_in_struc = triads["111U"][struc].nodes()
        neurons_in_111U.append(list(nodes_in_struc))

    nr_min_one_111U = len(np.unique(neurons_in_111U))
    rand_neurons_in_111U.append(len(triads["111U"]))
    ran_nr_min_one_111U.append(nr_min_one_111U)

    neurons_in_111D = []
    for struc in np.arange(len(triads["111D"])):
        nodes_in_struc = triads["111D"][struc].nodes()
        neurons_in_111D.append(list(nodes_in_struc))

    nr_min_one_111D = len(np.unique(neurons_in_111D))
    rand_neurons_in_111D.append(len(triads["111D"]))
    ran_nr_min_one_111D.append(nr_min_one_111D)

    neurons_in_120U = []
    for struc in np.arange(len(triads["120U"])):
        nodes_in_struc = triads["120U"][struc].nodes()
        neurons_in_120U.append(list(nodes_in_struc))

    nr_min_one_120U = len(np.unique(neurons_in_120U))
    rand_neurons_in_120U.append(len(triads["120U"]))
    ran_nr_min_one_120U.append(nr_min_one_120U)

    neurons_in_120D = []
    for struc in np.arange(len(triads["120D"])):
        nodes_in_struc = triads["120D"][struc].nodes()
        neurons_in_120D.append(list(nodes_in_struc))

    nr_min_one_120D = len(np.unique(neurons_in_120D))
    rand_neurons_in_120D.append(len(triads["120D"]))
    ran_nr_min_one_120D.append(nr_min_one_120D)

    neurons_in_120C = []
    for struc in np.arange(len(triads["120C"])):
        nodes_in_struc = triads["120C"][struc].nodes()
        neurons_in_120C.append(list(nodes_in_struc))

    nr_min_one_120C = len(np.unique(neurons_in_120C))
    rand_neurons_in_120C.append(len(triads["120C"]))
    ran_nr_min_one_120C.append(nr_min_one_120C)

    neurons_in_201 = []
    for struc in np.arange(len(triads["201"])):
        nodes_in_struc = triads["201"][struc].nodes()
        neurons_in_201.append(list(nodes_in_struc))

    nr_min_one_201 = len(np.unique(neurons_in_201))
    rand_neurons_in_201.append(len(triads["201"]))
    ran_nr_min_one_201.append(nr_min_one_201)


    neurons_in_210 = []
    for struc in np.arange(len(triads["210"])):
        nodes_in_struc = triads["210"][struc].nodes()
        neurons_in_210.append(list(nodes_in_struc))

    nr_min_one_210 = len(np.unique(neurons_in_210))
    rand_neurons_in_210.append(len(triads["210"]))
    ran_nr_min_one_210.append(nr_min_one_210)

    neurons_in_300 = []
    for struc in np.arange(len(triads["300"])):
        nodes_in_struc = triads["300"][struc].nodes()
        neurons_in_300.append(list(nodes_in_struc))

    nr_min_one_300 = len(np.unique(neurons_in_300))
    rand_neurons_in_300.append(len(triads["300"]))
    ran_nr_min_one_300.append(nr_min_one_300)

    neurons_in_030C = []
    for struc in np.arange(len(triads["030C"])):
        nodes_in_struc = triads["030C"][struc].nodes()
        neurons_in_030C.append(list(nodes_in_struc))
    nr_min_one_030C = len(np.unique(neurons_in_030C))
    rand_neurons_in_030C.append(len(triads["030C"]))
    ran_nr_min_one_030C.append(nr_min_one_030C)

    print("done with fconn")
    #######ACONN


    triads = nx.triads_by_type(G_aconn_ER)

    neurons_in_021D = []
    for struc in np.arange(len(triads["021D"])):
        nodes_in_struc = triads["021D"][struc].nodes()
        neurons_in_021D.append(list(nodes_in_struc))

    nr_min_one_021D = len(np.unique(neurons_in_021D))
    rand_neurons_in_021D_aconn.append(len(triads["021D"]))
    ran_nr_min_one_021D_aconn.append(nr_min_one_021D)

    neurons_in_021U = []
    for struc in np.arange(len(triads["021U"])):
        nodes_in_struc = triads["021U"][struc].nodes()
        neurons_in_021U.append(list(nodes_in_struc))

    nr_min_one_021U = len(np.unique(neurons_in_021U))
    rand_neurons_in_021U_aconn.append(len(triads["021U"]))
    ran_nr_min_one_021U_aconn.append(nr_min_one_021U)

    neurons_in_021C = []
    for struc in np.arange(len(triads["021C"])):
        nodes_in_struc = triads["021C"][struc].nodes()
        neurons_in_021C.append(list(nodes_in_struc))

    nr_min_one_021C = len(np.unique(neurons_in_021C))
    rand_neurons_in_021C_aconn.append(len(triads["021C"]))
    ran_nr_min_one_021C_aconn.append(nr_min_one_021C)

    neurons_in_030T = []
    for struc in np.arange(len(triads["030T"])):
        nodes_in_struc = triads["030T"][struc].nodes()
        neurons_in_030T.append(list(nodes_in_struc))

    nr_min_one_030T = len(np.unique(neurons_in_030T))
    rand_neurons_in_030T_aconn.append(len(triads["030T"]))
    ran_nr_min_one_030T_aconn.append(nr_min_one_030T)

    neurons_in_111U = []
    for struc in np.arange(len(triads["111U"])):
        nodes_in_struc = triads["111U"][struc].nodes()
        neurons_in_111U.append(list(nodes_in_struc))

    nr_min_one_111U = len(np.unique(neurons_in_111U))
    rand_neurons_in_111U_aconn.append(len(triads["111U"]))
    ran_nr_min_one_111U_aconn.append(nr_min_one_111U)

    neurons_in_111D = []
    for struc in np.arange(len(triads["111D"])):
        nodes_in_struc = triads["111D"][struc].nodes()
        neurons_in_111D.append(list(nodes_in_struc))

    nr_min_one_111D = len(np.unique(neurons_in_111D))
    rand_neurons_in_111D_aconn.append(len(triads["111D"]))
    ran_nr_min_one_111D_aconn.append(nr_min_one_111D)

    neurons_in_120U = []
    for struc in np.arange(len(triads["120U"])):
        nodes_in_struc = triads["120U"][struc].nodes()
        neurons_in_120U.append(list(nodes_in_struc))

    nr_min_one_120U = len(np.unique(neurons_in_120U))
    rand_neurons_in_120U_aconn.append(len(triads["120U"]))
    ran_nr_min_one_120U_aconn.append(nr_min_one_120U)

    neurons_in_120D = []
    for struc in np.arange(len(triads["120D"])):
        nodes_in_struc = triads["120D"][struc].nodes()
        neurons_in_120D.append(list(nodes_in_struc))

    nr_min_one_120D = len(np.unique(neurons_in_120D))
    rand_neurons_in_120D_aconn.append(len(triads["120D"]))
    ran_nr_min_one_120D_aconn.append(nr_min_one_120D)

    neurons_in_120C = []
    for struc in np.arange(len(triads["120C"])):
        nodes_in_struc = triads["120C"][struc].nodes()
        neurons_in_120C.append(list(nodes_in_struc))

    nr_min_one_120C = len(np.unique(neurons_in_120C))
    rand_neurons_in_120C_aconn.append(len(triads["120C"]))
    ran_nr_min_one_120C_aconn.append(nr_min_one_120C)

    neurons_in_201 = []
    for struc in np.arange(len(triads["201"])):
        nodes_in_struc = triads["201"][struc].nodes()
        neurons_in_201.append(list(nodes_in_struc))

    nr_min_one_201 = len(np.unique(neurons_in_201))
    rand_neurons_in_201_aconn.append(len(triads["201"]))
    ran_nr_min_one_201_aconn.append(nr_min_one_201)

    neurons_in_210 = []
    for struc in np.arange(len(triads["210"])):
        nodes_in_struc = triads["210"][struc].nodes()
        neurons_in_210.append(list(nodes_in_struc))

    nr_min_one_210 = len(np.unique(neurons_in_210))
    rand_neurons_in_210_aconn.append(len(triads["210"]))
    ran_nr_min_one_210_aconn.append(nr_min_one_210)

    neurons_in_300 = []
    for struc in np.arange(len(triads["300"])):
        nodes_in_struc = triads["300"][struc].nodes()
        neurons_in_300.append(list(nodes_in_struc))

    nr_min_one_300 = len(np.unique(neurons_in_300))
    rand_neurons_in_300_aconn.append(len(triads["300"]))
    ran_nr_min_one_300_aconn.append(nr_min_one_300)

    neurons_in_030C = []
    for struc in np.arange(len(triads["030C"])):
        nodes_in_struc = triads["030C"][struc].nodes()
        neurons_in_030C.append(list(nodes_in_struc))
    nr_min_one_030C = len(np.unique(neurons_in_030C))
    rand_neurons_in_030C_aconn.append(len(triads["030C"]))
    ran_nr_min_one_030C_aconn.append(nr_min_one_030C)





print("done")