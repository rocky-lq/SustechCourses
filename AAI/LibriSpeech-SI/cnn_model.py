import torch.nn as nn


# This class defines a convolutional neural network for speaker identification
class CNNSpeakerIdentificationModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the first convolutional block
        self.bk1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )

        # Define the second convolutional block
        self.bk2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Define the third convolutional block
        self.bk3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Define the fourth convolutional block
        self.bk4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # Define the fifth convolutional block
        self.bk5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        # Define the sixth convolutional block
        self.bk6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        # Adaptive average pooling layer with output size of 1
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        # Linear layer with 256 input features and 250 output features
        self.fc = nn.Linear(in_features=256, out_features=250)

    def forward(self, x):
        # Run the input through the convolutional blocks
        x = self.bk1(x)
        x = self.bk2(x)
        x = self.bk3(x)
        x = self.bk4(x)
        x = self.bk5(x)
        x = self.bk6(x)

        # Adaptive pool and flatten the output for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Run the output through the linear layer
        x = self.fc(x)

        # Return the final output
        return x
