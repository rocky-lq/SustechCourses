import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target
# print(X.shape, y.shape)
# print(X[:3])
# print(y[:3])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=3)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
# print(X_train[:3])
# print(y_train[:3])

import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


model = Model(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
print(model)

import tqdm

EPOCHS = 100
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()

X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

loss_list = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))

for epoch in tqdm.trange(EPOCHS):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss_list[epoch] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        y_pred = model(X_test)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)
ax1.plot(accuracy_list)
ax1.set_ylabel("Validation accuracy")

ax2.plot(loss_list)
ax2.set_ylabel("Validation loss")
ax2.set_xlabel("epochs")
fig.show()


def svm_sgd(X, Y):
    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 100000
    for epoch in range(1, epochs):
        for i, x in enumerate(X):
            if (Y[i] * np.dot(X[i], w)) < 1:
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1 / epoch) * w))
            else:
                w = w + eta * (-2 * (1 / epoch) * w)

    return


w = svm_sgd(X_train, y_train)
print(w)
