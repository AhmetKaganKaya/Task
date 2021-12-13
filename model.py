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



    def forward(self, sample):
        output = F.relu(self.fc1(sample))
        output = self.dropout1(output)
        output = F.relu(self.fc2(output))
        output = self.dropout2(output)
        output = torch.tanh(self.out(output))

        return output

class MLPModule(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(MLPModule, self).__init__()
        self.mlp = MLP(hidden_dim=hidden_dim ,input_dim=input_dim)

    def forward(self, query, sample):
        output = self.mlp(sample)
        output = torch.sigmoid(torch.matmul(output, query.unsqueeze(-1)))

        return output


