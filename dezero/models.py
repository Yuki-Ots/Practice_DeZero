from dezero import Layer
from dezero import utils
import dezero.layers as L
import dezero.functions as F
import math


def summary(model):
    line_width = 60
    doc = '😸Model Summary:\ntype: ' + model.__class__.__name__ + '\n'
    doc += 'Layer\t\tParameter\n' + '=' * line_width + '\n'
    sum_of_params = 0
    for l in model.layers:
        doc += l.__class__.__name__ + '\t'
        is_first_param = True
        for param in l.params():
            buffer = '\t' if is_first_param else '\t\t\t'
            doc += buffer + str(param.shape)
            doc += '\n'
            sum_of_params += math.prod(param.shape)
            is_first_param = False
        doc += '_' * line_width + '\n'
    doc += f'total parameter: {sum_of_params}'
    return doc

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

    def summary(self):
        return summary(self)


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, f'l{i}', layer)
            self.layers.append(layer)

    def forward(self, x):
        # 一番最後の要素はのぞく
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
