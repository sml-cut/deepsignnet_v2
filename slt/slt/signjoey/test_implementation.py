import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from layers import DenseBayesian
from utils import parameterConstraints, model_kl_divergence_loss
#torch.manual_seed(1000.)
#%matplotlib inline


def draw_plot(predicted, X, Y) :
    fig = plt.figure(figsize = (16, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    z1_plot = ax1.scatter(X[:, 0], X[:, 1], c = Y)
    z2_plot = ax2.scatter(X[:, 0], X[:, 1], c = predicted)

    plt.colorbar(z1_plot,ax=ax1)
    plt.colorbar(z2_plot,ax=ax2)

    ax1.set_title("REAL")
    ax2.set_title("PREDICT")

    plt.show()

iris = datasets.load_iris()

X = iris.data
Y = iris.target

x, y = torch.from_numpy(X).float(), torch.from_numpy(Y).long()

model = nn.Sequential(
    DenseBayesian(input_features=4, output_features=100, competitors = 2, activation = 'lwta',
                  prior_mean=0, prior_scale=1. ),
    DenseBayesian(input_features=100, output_features=3, competitors = 1, activation = 'linear',
                  prior_mean=0, prior_scale=1. ),
)

ce_loss = nn.CrossEntropyLoss()
kl_loss = model_kl_divergence_loss

optimizer = optim.Adam(model.parameters(), lr=0.1)

kl_weight = 0.01
pre = None
ce = None
constraints = parameterConstraints()

for step in range(3000):
    pre = model(x)
    ce = ce_loss(pre, y)
    kl = kl_loss(model)
    cost = ce + kl_weight * kl

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    model.apply(constraints)


model_children = list(model.children())
for layer in model_children:
    layer.deterministic = True

_, predicted = torch.max(pre.data, 1)
total = y.size(0)
correct = (predicted == y).sum()
print('- Accuracy: %f %%' % (100 * float(correct) / total))
print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item())) #kl.item

pre = model(x)
_, predicted = torch.max(pre.data, 1)
