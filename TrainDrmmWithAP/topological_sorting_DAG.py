from toposort import toposort, toposort_flatten

DAGFILE='/store/causalIR/model-aware-qpp/directed_graph.edges.acyclic'

with open(DAGFILE) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

adjmap = {}

for e in content:
    nodes = e.split()
    source = nodes[0]
    sink = nodes[1]
    if not source in adjmap:
        adjmap[source] = []
    adjmap[source].append(sink)
#print (adjmap)
for n in adjmap.keys():
    adjmap[n] = set(adjmap[n])
    #print ('{} {}'.format(n, adjmap[n]))

order = list(toposort(adjmap))
# print('1st : ', order)

order = list(reversed(order))
# print('\n2nd : ', order)

num = 0
# print(len(adjmap.keys()) + 1)
denom = len(adjmap.keys()) + 1
for x in order:
    x = sorted(list(x))
    for y in x:
        print(y, '\t', num/denom)
        num += 1
