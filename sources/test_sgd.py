import numpy as np
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

np.random.seed(3)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.randn(100, 1)

x = Variable(x)
y = Variable(y)

lr = 0.2
max_iter = 1000
hidden_size = 10

optims = [optimizers.SGD(lr), optimizers.Momentum(lr), optimizers.Adam()]
optim_names = ['SGD', 'Momentum', 'Adam']
losses_record = []


for optimizer in optims:
    model = MLP((hidden_size, 1))
    optimizer.setup(model)
    losses = []

    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y_pred, y)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        if i % 10 == 0:
            losses.append(loss.data)

    losses_record.append(losses)

for i in range(len(losses_record)):
    plt.plot(losses_record[i], label=optim_names[i])

plt.legend()
plt.show()
