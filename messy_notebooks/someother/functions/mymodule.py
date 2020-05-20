import numpy as np
from .someother import printer


def calc(x, y):
    return np.mean((np.mean(x) - y)**2)


class MyClass:
    
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        
    def mycalc(self):
        return calc(self.a, self.b)
    
    def __str__(self):
        return printer(self)