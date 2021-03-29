# this program inputs a graph in the form qid\tqid\tpredicted_class(CNN weights in [0,1])
# creates a directed graph based on the CNN predicted class labels(>= 0.5 --> class1 and < 0.5 --> class0)

import csv
import os


predict_file = open('/store/causalIR/model-aware-qpp/test_with_GT/gt.pair.1000.txt', 'r')
read_predict_file = predict_file.readlines()
out_graph_file = open('/store/causalIR/model-aware-qpp/test_with_GT/directed_graph.1000.edges', 'w')

for line in read_predict_file:
    parts = line.split('\t')
    if float(parts[2]) == 0:
        out_graph_file.writelines(parts[0] + ' ' + parts[1] + '\n')
    else:
        out_graph_file.writelines(parts[1] + ' ' + parts[0] + '\n')
out_graph_file.close()