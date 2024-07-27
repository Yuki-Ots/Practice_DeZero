import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero.dataloaders import DataLoader

max_epoch = 19
batch_size = 30
hidden_size = 10
lr = 1.0

x_orig, t_orig = dezero.datasets.get_spiral(train=True)
train_set = dezero.datasets.Spiral(train=True)
val_set = dezero.datasets.Spiral(train=False)
train_loadear = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

model = MLP((hidden_size, 3), activation=F.relu)
optimizer = optimizers.MomentumSGD().setup(model)
data_size = len(train_set)

train_losses = []
val_losses = []
folder = 'anime2'


# anim_list = []

def show(epoch, losses, val_losses, max_epoch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    h = 0.001
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X = np.c_[xx.ravel(), yy.ravel()]

    with dezero.no_grad():
        score = model(X)
    predict_cls = np.argmax(score.data, axis=1)
    Z = predict_cls.reshape(xx.shape)
    axes[0].contourf(xx, yy, Z)
    axes[1].set_xlim([0, max_epoch])
    axes[1].plot(losses, label='train loss', marker="o")
    axes[1].plot(val_losses, label='val loss', marker="o")
    axes[1].legend()
    axes[0].set_title('loss and epoch')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')

    # Plot data points of the dataset
    N, CLS_NUM = 100, 3
    markers = ['o', 'x', '^']
    colors = ['red', 'blue', 'green']
    for i in range(len(x_orig)):
        c = t_orig[i]
        axes[0].scatter(x_orig[i][0], x_orig[i][1], s=40, marker=markers[c], c=colors[c])
    # plt.savefig(f'{folder}/anime_{epoch}.png')


for epoch in range(max_epoch):
    print(f'{epoch} / {max_epoch}')
    index = np.random.permutation(data_size)
    sum_loss = 0
    val_loss = 0
    for x, t in train_loadear:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += loss.data

    for val_x, val_t in val_loader:
        with dezero.test_mode():
            y_hat = model(val_x)
            loss = F.softmax_cross_entropy(y_hat, val_t)
            val_loss += loss.data

    # Print loss every epoch
    train_avg_loss = sum_loss / len(train_set)
    train_losses.append(train_avg_loss)
    val_avg_loss = val_loss / len(val_set)
    val_losses.append(val_avg_loss)
    show(epoch, train_losses, val_losses, max_epoch)
