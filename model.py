import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MLP(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, input_dim)


    def forward(self, sample, query):
        # output = F.relu(self.bn1(self.fc1(sample)))
        # output = self.dropout1(output)
        # output = F.relu(self.bn2(self.fc2(output)))
        # output = self.dropout2(output)
        # output = torch.tanh(self.out(output))
        output = self.bn1(self.conv1(sample))
        output = self.bn2(self.conv2(output))
        output = output.reshape(output.size(0), -1)
        output = torch.tanh(self.out(output))

        output = torch.sigmoid(torch.matmul(query.reshape(query.shape[0], query.shape[1], -1), output.unsqueeze(2)))


        return output

class Module():
    pass

