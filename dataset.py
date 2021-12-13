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
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomRotation(degrees=(0, 45)),
            transforms.ToTensor(),
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

class MNISTDataset(datasets.VisionDataset):
    def __init__(self, most_digit, query_size, type, train=True):
        super(MNISTDataset, self).__init__(most_digit)
        mnist = datasets.MNIST(root= "input/", download= True, train= True)
        all_labels = mnist.targets
        if train:
            all_data = transform_train_images(mnist.data[torch.where(all_labels <= most_digit)[0]])
            self.labels = all_labels[torch.where(all_labels <= most_digit)[0]]
        else:
            all_data = transform_test_images(mnist.data[torch.where(all_labels > most_digit)[0]])
            self.labels = all_labels[torch.where(all_labels > most_digit)[0]]

        if type == 'mlp':
            self.dataset = all_data.reshape(-1,784).type(torch.FloatTensor)
        elif type == 'cnn':
            self.dataset = all_data.unsqueeze(1).type(torch.FloatTensor)


        self.number = query_size

    def __getitem__(self, index):
        query_label = torch.cat((torch.ones(self.number // 2), torch.zeros(self.number // 2)), dim=0)
        class_sample = self.dataset[index]
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
        query_label = query_label[shuffle_idx]

        return query, query_label, class_sample

    def __len__(self):
        return self.dataset.shape[0]

if __name__ == "__main__":
    train_dataset = MNISTDataset(5, 8, True)
    test_dataset = MNISTDataset(5,8, False)
    # for i in range(2):
    #     for j in range(4):
    #         plt.subplot(2,4, i*4+j+1)
    #         plt.imshow(np.array(query[i*4+j]).reshape(28,28), cmap='gray')
    #
    # print(target)
    # plt.show()
    print("a")