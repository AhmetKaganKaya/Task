from train import train
from model import MLP
from dataset import DigitDataset
import torch
import torch.nn as nn
import torch.optim as optim
import argparse


parser = argparse.ArgumentParser(description='One-Shot MNIST Classification')
parser.add_argument('--num_iter', type=int, default=1000, metavar='B',
                    help='Number of iterations')
parser.add_argument('--test-iter', type=int, default=100, metavar='B',
                    help='Number of iterations')
parser.add_argument('--hidden-dim', type=int, default=128, metavar='B',
                    help='Number of neurons in hidden layer')
parser.add_argument('--query-size', type=int, default=16, metavar='B',
                    help='Query size')
parser.add_argument('--lr', type=float, default=1e-4, metavar='B',
                    help='Learning Rate')
parser.add_argument('--save', type=str, default='/output/', metavar='B',
                    help='board dir')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
dataset = DigitDataset(query_size=args.query_size, transform=None)
model = MLP(args.hidden_dim, dataset.data.shape[-1]).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
bce_loss = nn.BCELoss()

if __name__ == "__main__":
    train_set = [0, 1, 2, 3, 4, 5]
    test_set = [6, 7, 8, 9]
    train(args=args, dataset=dataset, model=model, optimizer=optimizer, loss=bce_loss, device=device, train_set=train_set, test_set=test_set)
