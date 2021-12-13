from train import train
from model import MLPModule, CNNModule
from dataset import MNISTDataset
from utils import setLogger, visualize_result
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse

parser = argparse.ArgumentParser(description='One-Shot MNIST Classification')
parser.add_argument('--num_iter', type=int, default=5000, metavar='B',
                    help='Number of epoch')
parser.add_argument('--train-iter', type=int, default=100, metavar='B',
                    help='Number of train iterations')
parser.add_argument('--test-iter', type=int, default=50, metavar='B',
                    help='Number of test iterations')
parser.add_argument('--hidden-dim', type=int, default=128, metavar='B',
                    help='Number of neurons in hidden layer')
parser.add_argument('--image-size', type=int, default=28, metavar='B',
                    help='Width and Height')
parser.add_argument('--query-size', type=int, default=8, metavar='B',
                    help='Query size')
parser.add_argument('--module-type', type=str, default='cnn' , metavar='B',
                    help='Module Type: MLP or CNN')
parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                    help='Batch size of queries')
parser.add_argument('--lr', type=float, default=3e-5, metavar='B',
                    help='Learning Rate')
parser.add_argument('--save', type=str, default='output/', metavar='B',
                    help='outputs path')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
if not os.path.exists(args.save):
    os.mkdir(os.path.join(args.save))
if not os.path.exists(os.path.join(args.save, "checkpoint")):
    os.mkdir(os.path.join(args.save, "checkpoint"))
logFile = os.path.join(args.save, args.module_type +  '_results.log')
logger = setLogger(logFile)
logger.info(args)
logger.info("Experiment Starts!")
train_dataset = MNISTDataset(query_size=args.query_size, most_digit=5, type=args.module_type, train=True)
test_dataset = MNISTDataset(query_size=args.query_size, most_digit=5, type=args.module_type, train=False)
if args.module_type == 'mlp':
    model = MLPModule(args.hidden_dim, args.image_size * args.image_size).to(device)
elif args.module_type == 'cnn':
    model = CNNModule(args.image_size).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay= 1e-4)
bce_loss = nn.BCELoss()

if __name__ == "__main__":
    train_loss, test_accuracy = train(args=args, train_dataset=train_dataset, test_dataset=test_dataset, model=model, optimizer=optimizer, loss=bce_loss, logger= logger, device=device)
    dict = {"Train Loss": train_loss, "Test Accuracy": test_accuracy}
    visualize_result(args, dict)