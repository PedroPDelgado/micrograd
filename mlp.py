import random
from value import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x):
        dot_b = sum((xi*wi for xi, wi in zip(x, self.w)), start=self.b)
        out = dot_b.tanh()
        return out

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def parameters(self):
        return [param for n in self.neurons for param in n.parameters()]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, layers):
        self.layers = [Layer(layers[i], layers[i+1]) for i in range(len(layers)-1)]

    def parameters(self):
        return [param for l in self.layers for param in l.parameters()]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def backward(self):
        pass

if __name__ == "__main__":

    X = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    mlp = MLP((3, 4, 4, 1))

    y_preds = [mlp(x) for x in X]
    from loss import rmse
    loss = rmse(ys, y_preds)
    loss.backward()
    print(len(mlp.parameters()))
    print(mlp.layers[0].neurons[0].w[0].grad)

    """from graph_visualizer import draw_dot
    dot = draw_dot(loss)
    dot.render("mlp_rmse", view=True)
    """