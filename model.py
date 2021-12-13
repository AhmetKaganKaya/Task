import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, sample):
        output = F.relu(self.fc1(sample))
        output = self.dropout1(output)
        output = F.relu(self.fc2(output))
        output = self.dropout2(output)
        output = torch.tanh(self.out(output))

        return output

class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        cnn_out = self.conv2(self.conv1(torch.randn(1,1,input_dim, input_dim))).reshape(1,-1).shape[-1]
        self.out = nn.Linear(cnn_out, input_dim * input_dim)

    def forward(self, sample):
        output = self.conv1(sample)
        output = self.conv2(output)
        output = output.reshape(output.shape[0], -1)
        output = self.out(output)

        return output

class MLPModule(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(MLPModule, self).__init__()
        self.mlp = MLP(hidden_dim=hidden_dim ,input_dim=input_dim)

    def forward(self, sample, query):
        output = self.mlp(sample)
        output = torch.sigmoid(torch.matmul(query, output.unsqueeze(-1)))

        return output

class CNNModule(nn.Module):
    def __init__(self, input_dim):
        super(CNNModule, self).__init__()
        self.cnn = CNN(input_dim)

    def forward(self, sample, query):
        output = self.cnn(sample)
        query = query.flatten(start_dim = 2)
        output = torch.sigmoid(torch.matmul(query, output.unsqueeze(-1)))

        return output




