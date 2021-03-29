import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
import networkx as nx


class GraphEdgeInstance:
    def __init__(self, line):
        l = line.strip().split('\t')
        self.qid_a = l[0]
        self.qid_b = l[1]
        self.confidence = float(l[4])

class PairedInstanceNodes:
    '''
    Each line in this file should comprise three tab separated fields
    <id1> <id2> <confidence value>
    '''

    def __init__(self, idpairConfidenceFile):
        self.graph = {}

        with open(idpairConfidenceFile) as f:
            content = f.readlines()

        for x in content:
            instance = GraphEdgeInstance(x)
            # print(instance.qid_a)
            if instance.qid_a not in self.graph.keys():
                # print('**********')
                qid_list = []
                qid_list.append(instance)
                self.graph[instance.qid_a] = qid_list
            else:
                # print('###########')
                qid_list.append(instance)
                self.graph[instance.qid_a] = qid_list

allPairs_graph = PairedInstanceNodes('/store/causalIR/model-aware-qpp/confidence_graph.weights')
# print(allPairs_graph.graph)

res_file = open('/store/causalIR/model-aware-qpp/confidence.sparse', 'w')
start_qid = int(list(allPairs_graph.graph.keys())[0]) - 1
# print('START : ', start_qid)
end_qid = int(list(allPairs_graph.graph.keys())[0]) + len(allPairs_graph.graph)
# print('END : ', end_qid)

for node in allPairs_graph.graph.keys():
    # print('NODE : ', start_qid)
    count = int(node) - start_qid
    while count > 0:
        res_file.writelines('0.0' + ' ')
        count = count - 1
    for edge in allPairs_graph.graph[node]:
        res_file.writelines(str(edge.confidence) + ' ')
    res_file.writelines('\n')

cnt = end_qid - start_qid
while cnt > 0:
    res_file.writelines('0.0' + ' ')
    cnt = cnt - 1
res_file.close()

# =================================================

# read_sparse = np.genfromtxt('/store/causalIR/model-aware-qpp/confidence.sparse', delimiter=" ")
# sparse_matrix = csr_matrix(read_sparse[:, :])
# # print(sparse_matrix[0])
# Tcsr = minimum_spanning_tree(sparse_matrix)
# print(Tcsr)

# ================================================

read_sparse = np.genfromtxt('/store/causalIR/model-aware-qpp/confidence.sparse', delimiter=" ")
G = nx.from_numpy_array(np.array(read_sparse[:, :]))
print(G.nodes())
pos = dict( (n, n) for n in G.nodes() )
print(pos)
labels = dict( ((i, j), i + (int(len(G))-1-j) * int(len(G)) ) for i, j in G.nodes() )
print(labels)
# nx.relabel_nodes(G,labels,False)
# inds=labels.keys()
# vals=labels.values()
# inds.sort()
# vals.sort()
# pos2=dict(zip(vals,inds))
# nx.draw_networkx(G, pos=pos2, with_labels=False, node_size = 15)




