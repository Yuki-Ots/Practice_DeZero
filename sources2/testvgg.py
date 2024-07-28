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



class VGG16(models.Model):
    def __init__(self):
        super().__init__()
        self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(10)


    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling_simple(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling_simple(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling_simple(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling_simple(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling_simple(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x



batch_size = 100
max_epoch = 30


print(f'{dezero.cuda.gpu_enable=}')
train_set = datasets.CIFAR10(train=True)
val_set = datasets.CIFAR10(train=False)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, gpu=dezero.cuda.gpu_enable)
val_dataloader = DataLoader(val_set, batch_size, gpu=dezero.cuda.gpu_enable)
model = VGG16()
optimizer = optimizers.Momentum().setup(model)
loss_list = []
val_loss_list = []
acc_list = []

batch_size = 1000
max_epoch = 10
count = 0

for epoch in tqdm.tqdm(range(max_epoch)):
    sum_loss = 0.
    val_sum_loss = 0.
    acc = 0
    for x, t in dataloader:
        count += 1
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += loss.data
    with dezero.test_mode():
        for val_x, val_t in val_dataloader:
            y = model(val_x)
            loss = F.softmax_cross_entropy(y, val_t)
            val_sum_loss += loss.data
            t_hat = np.argmax(y.data, axis=-1)
            acc += np.sum(t_hat == val_t)

    loss_list.append((sum_loss / len(train_set)).get())
    val_loss_list.append((val_sum_loss / len(val_set)).get())
    acc_list.append((acc / len(val_set)).get())
