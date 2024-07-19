import dezero
import dezero.functions as F
import dezero.layers as L
import numpy as np
import dezero.utils as utils
from dezero import Layer

model = Layer()
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)

def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y

for p in model.params():
    print(p)

class TwoLayerNet(Layer):
    def __init__(self, hidden_size, output_size):
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(output_size)

    def forward(self, x):
        x = self.l1(x)
        x = F.sigmoid(x)
        x = self.l2(x)
        return x
