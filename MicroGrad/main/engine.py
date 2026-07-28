
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
            self.grad += 
            other.grad += 