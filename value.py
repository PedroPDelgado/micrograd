import numpy as np
from utils import build_topo

class Value:
    def __init__(self, data, children=(), op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op
        self._label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def exp(self):
        exp = np.exp(self.data)
        out = Value(exp, (self, ), "exp")
        def _backward():
            self.grad += exp * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other): # self/other
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        v = self.data ** other
        out = Value(v, (self,), f"**{other}")
        def _backward():
            self.grad += other * self.data ** (other-1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        tanh = (np.exp(2*self.data) - 1) / (np.exp(2*self.data) + 1)
        out = Value(tanh, (self,), "tanh")
        def _backward():
            self.grad += (1 - tanh ** 2) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def backward(self):
        self.grad = 1
        topo = build_topo(self)
        for v in topo[::-1]:
            v._backward()
