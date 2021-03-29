import pydot

# menu = {'dinner': {'chicken':'good', 'beef':'average', 'vegetarian':{
#                    'tofu':'good',
#                    'salad':{
#                             'caeser':'bad',
#                             'italian':'average'}
#                    },
#              'pork':'bad'}
#         }

menu = {'401': {'402': 0.8559, '403': 0.86917, '405': 0.86545, '406': 0.90007, '407': 0.86739, '408': 0.90403,
                '411': 0.92052, '412': 0.83803, '414': 0.78632, '415': 0.81221, '418': 0.8324, '419': 0.83111,
                '420': 0.78542}, '403': {'404': 0.89061, '409': 0.92908, '410': 0.94373}, '412': {'413': 0.91189},
                '414': {'416': 0.78931, '417': 0.83039}, '420': {'421': 0.78484, '422': 0.80178, '424': 0.79689,
                '426': 0.80466, '428': 0.7912, '429': 0.79256, '431': 0.81763, '432': 0.79332, '433': 0.81391,
                '435': 0.78569, '436': 0.80351, '437': 0.79072, '438': 0.79047, '439': 0.79191, '440': 0.79508,
                '441': 0.81751, '442': 0.83179, '443': 0.78824, '444': 0.79367, '446': 0.82742, '447': 0.79447,
                '448': 0.81075, '449': 0.79449}, '421': {'423': 0.79793, '425': 0.78654, '427': 0.81661,
                '430': 0.80624, '434': 0.79203}, '435': {'445': 0.81694, '450': 0.79705}}


def draw(parent_name, child_name):
    print('amal : ', parent_name)
    print('tunu : ', child_name)
    edge = pydot.Edge(parent_name, child_name)
    print('EDGE : ', edge)
    graph.add_edge(edge)
    print('GRAPH : ', graph)

def visit(node, parent=None):
    for k,v in node.items():
        print('K : ', k)
        print('V : ', v)
        if isinstance(v, dict):
            print('@@@@@@')
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                print('%%%%%')
                draw(parent, k)
            visit(v, str(k))
        else:
            print('PARENT : ', parent)
            print('K : ', k)
            draw(parent, k)
            # drawing the label using a distinct name
            draw(str(k), str(v))

graph = pydot.Dot(graph_type='graph')
visit(menu)
graph.write_png('example1_graph.png')