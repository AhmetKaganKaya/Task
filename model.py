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



    def forward(self, sample, query):
        output = F.relu(self.bn1(self.fc1(sample)))
        output = self.dropout1(output)
        output = F.relu(self.bn2(self.fc2(output)))
        output = self.dropout2(output)
        output = torch.tanh(self.out(output))

        output = torch.sigmoid(torch.matmul(query.reshape(query.shape[0], query.shape[1], -1), output.unsqueeze(2)))


        return output

class Module():
    pass

