import torchvision.datasets as datasets
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random


class DigitDataset():
    def __init__(self, query_size, transform):
        self.mnist = datasets.MNIST(root="input/", train=True, download=True, transform=transform)
        self.data = self.mnist.data.reshape(self.mnist.data.shape[0], -1)/255
        self.labels = self.mnist.targets.type(torch.long)
        self.classes = torch.tensor(sorted(torch.unique(self.mnist.targets)))
        self.number = query_size
        self.transform = transform
        self.training_samples = self.set_training_samples()

    def get_batch(self, number):
        positive = self.data[torch.where(self.labels == number)[0]]
        positive = positive[torch.randperm(positive.shape[0])]
        # Shuffle
        negative = self.data[torch.where(self.labels != number)[0]]
        negative = negative[torch.randperm(negative.shape[0])]
        batches = []
        for i in range(positive.shape[0] // (self.number // 2)):
            label = torch.cat((torch.ones(self.number // 2), torch.zeros(self.number // 2)), dim=0)
            query_p = positive[i * (self.number // 2): (i + 1)* (self.number// 2), :]
            query_n = negative[np.random.choice(negative.shape[0], self.number // 2), :]
            # Merge them
            query = torch.cat((query_p,query_n), dim=0)
            shuffle_idx = torch.randperm(self.number)
            # Shuffle query
            query = query[shuffle_idx]
            label = label[shuffle_idx]
            batches.append((query, label, self.training_samples[number]))
        return batches

    def set_training_samples(self):
        samples = []
        for i in range(self.classes.shape[0]):
            idxs = torch.where(self.labels==i)[0]
            idx = idxs[random.randint(0,idxs.shape[0] - 1)]
            samples.append(self.data[idx])
        return samples

if __name__ == "__main__":
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    d = DigitDataset(8, transform=transform)
    for k in [0,1,2,3,4,5]:
        batches= d.get_batch(k)
        # print(data[1])
        for i in range(2):
            for j in range(4):
                plt.subplot(2,4,i*4 + j + 1)
                plt.imshow(np.array(batches[3][0][i*4 + j]).reshape(28,28), cmap='gray')
        plt.show()
        # print(label)


