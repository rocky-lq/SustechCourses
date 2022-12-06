import numpy as np
import torch
from torch import nn

# 参考 https://blog.csdn.net/qq_38316300/article/details/124815632
x_train = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

y_train = np.array([0, 1, 1, 0, 1, 0, 0, 1])


class ANN(nn.Module):
    # 输入3，输出2，包含一个隐藏层，隐藏层有6个神经元
    def __init__(self, num_classes=2):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(3, 6)
        self.fc2 = nn.Linear(6, num_classes)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train():
    # 损失函数
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # 将numpy数组转换为tensor
        inputs = torch.from_numpy(x_train).float()
        targets = torch.from_numpy(y_train).long()

        # 前向传播
        outputs = model(inputs)
        l = loss(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, l.item()))
            # 验证训练集的准确率
            correct = 0
            total = 0
            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            print('Accuracy of the network on the 8 train samples: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    batch_size = x_train.shape[0]
    model = ANN()
    num_epochs = 1000
    learning_rate = 0.01
    train()
