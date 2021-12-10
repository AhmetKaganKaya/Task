import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, sample, input):
        output = F.relu(self.fc1(sample))
        output = F.relu(self.fc2(output))
        output = torch.tanh(self.out(output))

        output = torch.sigmoid(torch.matmul(input, output.T).squeeze(1))

        return output


