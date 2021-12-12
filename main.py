from train import train
from model import MLP
from dataset import MNISTTrainDataset,MNISTTestDataset
import torch
import torch.nn as nn
import torch.optim as optim
import argparse


parser = argparse.ArgumentParser(description='One-Shot MNIST Classification')
parser.add_argument('--num_iter', type=int, default=5000, metavar='B',
                    help='Number of iterations')
parser.add_argument('--test-iter', type=int, default=100, metavar='B',
                    help='Number of iterations')
parser.add_argument('--hidden-dim', type=int, default=128, metavar='B',
                    help='Number of neurons in hidden layer')
parser.add_argument('--query-size', type=int, default=8, metavar='B',
                    help='Query size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='B',
                    help='Learning Rate')
parser.add_argument('--save', type=str, default='/output/', metavar='B',
                    help='outputs path')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
train_dataset = MNISTTrainDataset(query_size=args.query_size, most_digit=6)
test_dataset = MNISTTestDataset(query_size=args.query_size, least_digit=7)
model = MLP(args.hidden_dim, 784).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay= 1e-4)
bce_loss = nn.BCELoss()

if __name__ == "__main__":
    train(args=args, train_dataset=train_dataset, test_dataset=test_dataset, model=model, optimizer=optimizer, loss=bce_loss, device=device, batch_size=32)
