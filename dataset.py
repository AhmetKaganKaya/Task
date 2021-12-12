import torchvision.datasets as datasets
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def transform_train_images(dataset):
    for i in range(dataset.shape[0]):
        data = dataset[i]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomRotation(degrees=(0, 45)),
            transforms.Normalize((0.1307,), (0.3081,))])
        img = Image.fromarray(data.numpy(), mode='L')

        if transform is not None:
            img = transform(img)
        dataset[i] = img
    return dataset

def transform_test_images(dataset):
    for i in range(dataset.shape[0]):
        data = dataset[i]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        img = Image.fromarray(data.numpy(), mode='L')

        if transform is not None:
            img = transform(img)
        dataset[i] = img
    return dataset

class MNISTTrainDataset(datasets.VisionDataset):
    def __init__(self, most_digit, query_size):
        super(MNISTTrainDataset, self).__init__(most_digit)
        mnist = datasets.MNIST(root= "/input/", download= True, train= True)
        all_labels = mnist.targets
        all_data = transform_train_images(mnist.data[torch.where(all_labels <= most_digit)[0]])
        # self.dataset = all_data.reshape(-1,784) / 1.
        self.dataset = all_data.reshape(-1,1,28,28) / 1.
        self.labels = all_labels[torch.where(all_labels <= most_digit)[0]]

        self.number = query_size

    def __getitem__(self, index):
        target = torch.cat((torch.ones(self.number // 2), torch.zeros(self.number // 2)), dim=0)
        number = self.labels[index]
        positive = self.dataset[torch.where(self.labels == number)[0]]
        a = np.random.choice(positive.shape[0], self.number // 2)
        query_p = positive[a, :]
        negative = self.dataset[torch.where(self.labels != number)[0]]
        query_n = negative[np.random.choice(negative.shape[0], self.number // 2), :]
        query = torch.cat((query_p, query_n), dim=0)
        shuffle_idx = torch.randperm(self.number)
        # Shuffle query
        query = query[shuffle_idx]
        target = target[shuffle_idx]

        return query, target, self.dataset[index]

    def __len__(self):
        return self.dataset.shape[0]


class MNISTTestDataset(datasets.VisionDataset):
    def __init__(self, least_digit, query_size):
        super(MNISTTestDataset, self).__init__(least_digit)
        mnist = datasets.MNIST(root= "/input/", download= True, train= False)
        all_labels = mnist.targets
        test_data = transform_test_images(mnist.data[torch.where(all_labels >= least_digit)[0]])
        # self.dataset = test_data.reshape(-1,784) / 1.
        self.dataset = test_data.reshape(-1,1,28,28) / 1.

        self.labels = all_labels[torch.where(all_labels >= least_digit)[0]]

        self.number = query_size

    def __getitem__(self, index):
        target = torch.cat((torch.ones(self.number // 2), torch.zeros(self.number // 2)), dim=0)
        number = self.labels[index]
        positive = self.dataset[torch.where(self.labels == number)[0]]
        a = np.random.choice(positive.shape[0], self.number // 2)
        query_p = positive[a, :]
        negative = self.dataset[torch.where(self.labels != number)[0]]
        query_n = negative[np.random.choice(negative.shape[0], self.number // 2), :]
        query = torch.cat((query_p, query_n), dim=0)
        shuffle_idx = torch.randperm(self.number)
        # Shuffle query
        query = query[shuffle_idx]
        target = target[shuffle_idx]

        return (query, target, self.dataset[index])

    def __len__(self):
        return self.dataset.shape[0]

if __name__ == "__main__":
    dataset = MNISTTrainDataset(5, 8)
    query,target, sample, = dataset.__getitem__(2005)
    for i in range(2):
        for j in range(4):
            plt.subplot(2,4, i*4+j+1)
            plt.imshow(np.array(query[i*4+j]).reshape(28,28), cmap='gray')

    print(target)
    plt.show()
    print("a")