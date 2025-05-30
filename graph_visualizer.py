from graphviz import Digraph
from value import Value
from utils import build_topo

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    print("Nodes:", nodes)
    print("Edges:", edges)
    for n in nodes:
        uid = str(id(n))
        # for every value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{%s | data %.4f | grad %.4f }" % (n._label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

if __name__ == "__main__":
    """x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813736870195432, label="b")
    x1w1 = x1 * w1
    x1w1._label = "x1w1"
    x2w2 = x2 * w2
    x2w2._label = "x2w2"
    x1w1_p_x2w2 = x1w1 + x2w2
    x1w1_p_x2w2._label = "x1w1 + x2w2"
    n = x1w1_p_x2w2 + b
    n._label = "x1w1 + x2w2 + b"
    o = n.tanh()
    o._label = "tanh(x1w1 + x2w2 + b)"
    o.backward()"""

    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813736870195432, label="b")
    x1w1 = x1 * w1
    x1w1._label = "x1w1"
    x2w2 = x2 * w2
    x2w2._label = "x2w2"
    x1w1_p_x2w2 = x1w1 + x2w2
    x1w1_p_x2w2._label = "x1w1 + x2w2"
    n = x1w1_p_x2w2 + b
    n._label = "x1w1 + x2w2 + b"
    e = (2*n).exp()
    e._label = "e"
    o = (e - 1) / (e + 1)
    o._label = "o"
    o.backward()
    dot = draw_dot(o)
    dot.render('graph_output', view=True)

