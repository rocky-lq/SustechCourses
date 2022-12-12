import torch
import torch.nn as nn
import torch.nn.functional as F

from load_data import load_data


# 生成一个语音识别的模型，输出类别为250类，输入为200000个点的语音数据
class VoiceFilter(nn.Module):
    def __init__(self, num_classes=250):
        super(VoiceFilter, self).__init__()

        self.conv = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn2
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),  # (9, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn5
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)),  # (17, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn6
            nn.ZeroPad2d((2, 2, 16, 16)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)),  # (33, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn7
            nn.ZeroPad2d((2, 2, 32, 32)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)),  # (65, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn8
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            8 * 601 + 256,
            400,
            batch_first=True,
            bidirectional=True)

        self.fc1 = nn.Linear(2 * 400, 600)
        self.fc2 = nn.Linear(600, num_classes)

    def forward(self, x, dvec):
        # x: [B, T, num_freq]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_freq]
        x = self.conv(x)
        # x: [B, 8, T, num_freq]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, 8, num_freq]
        x = x.view(x.size(0), x.size(1), -1)
        # x: [B, T, 8*num_freq]

        # dvec: [B, emb_dim]
        dvec = dvec.unsqueeze(1)
        dvec = dvec.repeat(1, x.size(1), 1)
        # dvec: [B, T, emb_dim]

        x = torch.cat((x, dvec), dim=2)  # [B, T, 8*num_freq + emb_dim]

        x, _ = self.lstm(x)  # [B, T, 2*lstm_dim]
        x = F.relu(x)
        x = self.fc1(x)  # x: [B, T, fc1_dim]
        x = F.relu(x)
        x = self.fc2(x)  # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.sigmoid(x)
        return


model = VoiceFilter(250)
# 训练模型
train_loader, val_loader = load_data()


# 训练模型
def train(model, train_loader, val_loader, num_epochs, learning_rate):
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (audio, target) in enumerate(train_loader):
            # 前向传播
            outputs = model(audio)
            loss = criterion(outputs, target)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                         loss.item()))

        # 在验证集
        with torch.no_grad():
            correct = 0
            total = 0
            for audio, target in val_loader:
                outputs = model(audio)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    train(model, train_loader, val_loader, num_epochs=5, learning_rate=0.001)
