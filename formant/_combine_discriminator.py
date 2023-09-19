import torch
import torch.nn as nn

LEAK = 0.1


class CombineDiscriminator(nn.Module):
    def __init__(self):
        super(CombineDiscriminator, self).__init__()

        # CNN layers for processing the generated spectrogram
        self.conv1_spec = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_spec = nn.LeakyReLU(LEAK, inplace=True)
        self.pool1_spec = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_spec = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_spec = nn.LeakyReLU(LEAK, inplace=True)
        self.pool2_spec = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_spec = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu3_spec = nn.LeakyReLU(LEAK, inplace=True)
        self.pool3_spec = nn.MaxPool2d(kernel_size=2, stride=2)

        # CNN layers for processing the heatmap
        self.conv1_heat = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2)
        self.relu1_heat = nn.LeakyReLU(LEAK, inplace=True)
        self.pool1_heat = nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv2_heat = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)
        self.relu2_heat = nn.LeakyReLU(LEAK, inplace=True)
        self.pool2_heat = nn.MaxPool2d(kernel_size=4, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(42672, 128)
        self.relu_fc1 = nn.LeakyReLU(LEAK, inplace=True)
        self.fc2 = nn.Linear(128, 128)
        self.relu_fc2 = nn.LeakyReLU(LEAK, inplace=True)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, spectrogram, heatmap):
        # Process the generated spectrogram
        x_spec = self.pool1_spec(
            self.relu1_spec(self.conv1_spec(spectrogram.unsqueeze(1)))
        )  # Add channel dimension
        x_spec = self.pool2_spec(self.relu2_spec(self.conv2_spec(x_spec)))
        x_spec = self.pool3_spec(self.relu3_spec(self.conv3_spec(x_spec)))
        x_spec = x_spec.view(x_spec.size(0), -1)

        # Process the heatmap
        x_heat = self.pool1_heat(self.relu1_heat(self.conv1_heat(heatmap)))
        x_heat = x_heat.view(x_heat.size(0), -1)

        # Concatenate processed features
        x_combined = torch.cat((x_spec, x_heat), dim=1)

        # Fully connected layers
        x = self.relu_fc1(self.fc1(x_combined))
        x = self.relu_fc2(self.fc2(x))
        x = self.fc3(x)
        output = self.sigmoid(x)

        return output
