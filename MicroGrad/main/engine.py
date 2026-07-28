
class Value():
    """Stores a value and its gradient"""
    def __init__(self, data, children = (), _op = ""):
        self.data = data
        self.grad = 0.0

        #internal vars
        self._backward = lambda: None
        self._prev = set(children)
        self._op = _op
    
    def __add__(self, other):
        if isinstance(other, Value):
            other = other
        else:
            other = Value(other)
        
        out = Value(self.data + other.data, (self, other), _op = "+")

        def _backward():
            self.grad += out.grad
            out.grad += self.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        if isinstance(other, Value):
            other = other
        else:
            other = Value(other)
        
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += (out.grad * other.data)
            other.grad += (out.grad * self.data)
        out._backward = _backward
        return out
    
    def __pow__(self, raised):
        assert isinstance(raised, (int, float)), "Only support float and int for now"
        out = Value(self.data ** raised, (self,), f"^{raised}")

        def _backward():
            self.grad += (raised * (self.data**(raised - 1))) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        if self.data <= 0:
            out = Value(0)
        else:
            out = Value(self.data, (self, ), "Relu")
        
        def _backward():
            if self.data <=  0:
                self.grad += 0
            else:
                self.grad += out.grad
        out._backward = _backward
        return out

    def _backward(self):
        pass

    
    def __radd__(self, other):
        return self + other

    def __neg__(self): # -self
        return self * -1

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

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
