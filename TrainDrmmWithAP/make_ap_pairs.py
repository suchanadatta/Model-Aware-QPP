# prepare input for model-aware-qpp
# inputs = two query IDs (say q1, q2)
# output = binary class label - 1/0 (1 -> q1>q2; 0 -> q1<q2)
# output format - q1 \t q2 \t lass_label

import sys

if len(sys.argv) < 3:
    print('Needs 2 arguments - <qid \t AP file> <output file path>')
    exit(0)

arg_qid_ap_file = sys.argv[1]
arg_res_file_path = sys.argv[2]

res_file = open(arg_res_file_path + 'qid_ap.pairs.gt', 'w')

qid_ap_dict = {}

def make_qid_ap_dict(ap_file):
    fp = open(ap_file)
    for line in fp.readlines():
        parts = line.rstrip().split('\t')
        qid_ap_dict[parts[0]] = parts[1]
    print('DICT : ', qid_ap_dict)

make_qid_ap_dict(arg_qid_ap_file)

qid_list = qid_ap_dict.keys()
print('LIST : ', qid_list)

for id in qid_list:
    curr_qid = id
    for entry in qid_list:
        if entry > curr_qid:
            if qid_ap_dict[curr_qid] > qid_ap_dict[entry]:
                res_file.writelines(curr_qid + '\t' + entry + '\t' + '1\n')
            else:
                res_file.writelines(curr_qid + '\t' + entry + '\t' + '0\n')