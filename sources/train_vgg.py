import dezero
import numpy as np
import dezero.functions as F
import dezero.utils as utils
import dezero.layers as L
import dezero.models as models
import dezero.datasets as datasets
from dezero.dataloaders import DataLoader
import dezero.optimizers as optimizers
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import dezero.cuda
import os


class CNNModel(models.Model):

    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1_1 = L.Conv2d(32, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(32, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(32, kernel_size=3, stride=1, pad=1)
        self.fc1 = L.Linear(512)
        self.fc2 = L.Linear(512)


    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling_simple(x, 2, 2, 0)
        x = F.relu(self.conv2_1(x))
        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return x



batch_size = 100
max_epoch = 10


print(f'{dezero.cuda.gpu_enable=}')
train_set = datasets.MNIST(train=True)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, gpu=dezero.cuda.gpu_enable)
model = CNNModel()
optimizer = optimizers.Momentum().setup(model)
loss_list = []
acc_list = []

for epoch in tqdm.tqdm(range(max_epoch)):
    sum_loss = 0.
    count = 0
    for x, t in dataloader:
        count += 1
        print(count)
        x = x.reshape(batch_size, 1, 28, 28)
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += loss.data

    loss_list.append(sum_loss / len(train_set))


file_path = 'cnn_mnist.npz'
if os.path.exists(file_path):
    model.load_weights(file_path)
