import sys
import pandas as pd
from scipy import stats


if len(sys.argv) < 3:
    print('Needs 2 arguments - <ground truth> <predicted file>')
    exit(0)

actual = sys.argv[1]
predict = sys.argv[2]

# def read_file(file):
#     fp = open(file)
#     class_label = []
#     for line in fp.readlines():
#         parts = line.split('\t')
#         if len(parts) == 3:
#             class_label.append(parts[2])
#         elif len(parts) == 4:
#             class_label.append(parts[3])
#         else:
#             print('Check input files')
#     return class_label

def read_topo_file(file):
    fp = open(file)
    order = []
    for line in fp.readlines():
        order.append(line)
    return order

def read_file(file):
    fp = open(file)
    scores = []
    for line in fp.readlines():
        scores.append(float(line.split('\t')[1]))
    return scores


# actual_label = read_file(actual)
# predict_label = read_file(predict)

actual_label = read_file(actual)
predict_label = read_file(predict)

# spearman's rank co-relation
# Calculate the rank of x's
xranks = pd.Series(actual_label).rank()
print('##### : ', xranks)

# Caclulate the ranking of the y's
yranks = pd.Series(predict_label).rank()
print('##### : ', yranks)

corrs, _ = stats.spearmanr(actual_label, predict_label)
print("Spearman's Rank correlation: %.5f" % corrs)

# Calculating Kendall Rank correlation
corrk, _ = stats.kendalltau(actual_label, predict_label)
print('Kendall Rank correlation: %.5f' % corrk)


# import sys
# import numpy as np
#
# def read_topo_file(file):
#     fp = open(file)
#     order = []
#     for line in fp.readlines():
#         order.append(line)
#     return order
#
# def read_file(file):
#     fp = open(file)
#     scores = []
#     for line in fp.readlines():
#         scores.append(float(line.split('\t')[1]))
#     return scores
#
# def normalised_kendall_tau_distance(values1, values2):
#     """Compute the Kendall tau distance."""
#     n = len(values1)
#     assert len(values2) == n, "Both lists have to be of equal length"
#     i, j = np.meshgrid(np.arange(n), np.arange(n))
#     # a = np.argsort(values1)
#     # b = np.argsort(values2)
#     a  = np.array(values1)
#     b = np.array(values2)
#     ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
#     return ndisordered / (n * (n - 1))
#
# if len(sys.argv) < 3:
#     print('Needs 2 arguments - <ground truth> <predicted file>')
#     exit(0)
#
# actual = sys.argv[1]
# predict = sys.argv[2]
#
# actual_label = read_file(actual)
# predict_label = read_file(predict)
#
# tau = normalised_kendall_tau_distance(actual_label, predict_label)
# print('distance : ', tau)




