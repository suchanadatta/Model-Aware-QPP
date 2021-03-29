import csv
import edmonds
from toposort import toposort, toposort_flatten


# Below, g is graph representation of minimum spanning tree
# root is the starting node of the MST, and G is the input graph

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
                qid_neighbor_dict = {}
                qid_neighbor_dict[instance.qid_b] = instance.confidence
                self.graph[instance.qid_a] = qid_neighbor_dict
            else:
                # print('###########')
                qid_neighbor_dict[instance.qid_b] = instance.confidence
                self.graph[instance.qid_a] = qid_neighbor_dict

allPairs_graph = PairedInstanceNodes('/store/causalIR/model-aware-qpp/confidence_graph.weights')
# print(allPairs_graph.graph)
min_arborescence = edmonds.mst("401", allPairs_graph.graph)
print(min_arborescence)

topo_dict = {}
for key in min_arborescence:
    topo_dict[key] = set(min_arborescence[key].keys())
print('TOPO DICT : ', topo_dict)
order = list(toposort(topo_dict))
print (order)

