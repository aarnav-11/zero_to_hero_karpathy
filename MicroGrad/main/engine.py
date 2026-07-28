
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
        