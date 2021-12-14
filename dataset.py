import torchvision.datasets as datasets
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def transform_train_images(dataset):
    '''
    This method takes train dataset and applied train transform
    Args:
        dataset: MNIST train dataset

    Returns:
        dataset: Transformed MNIST train dataset
    '''
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
    '''
    This method takes test dataset and applied test transform
    Args:
        dataset: MNIST test dataset

    Returns:
        dataset: Transformed MNIST test dataset
    '''
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
        '''
        Custom VisionDataset class.
        Args:
            most_digit: it splits dataset to train and test part. FOr example if most_digit = 5,
             train digits will be [0,1,2,3,4,5] and test digits will be [6,7,8,9]
            query_size: Query Size
            type: If model type is mlp it returns dimension of (1,784). If model type is CNN it return dimension of (1,28,28)
            train: Whether dataset is train or test
        '''
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
        '''
        This method get indexed image an make it target sample. Query is created according to label of the class of target sample.
        Args:
            index: Target sample is sampled from dataset

        Returns:
            query: Query list. Dimension is (query size, 784) or (query size, 1, 28, 28)
            query_label: Binary labels of query
            class_sample: Target sample
        '''
        query_label = torch.cat((torch.ones(self.number // 2), torch.zeros(self.number // 2)), dim=0)
        # Target sample is sampled
        class_sample = self.dataset[index]
        # Class of the target sample is obtained
        number = self.labels[index]
        # Positive and negative examples are sampled and concatenate
        positive = self.dataset[torch.where(self.labels == number)[0]]
        a = np.random.choice(positive.shape[0], self.number // 2)
        query_p = positive[a, :]
        negative = self.dataset[torch.where(self.labels != number)[0]]
        query_n = negative[np.random.choice(negative.shape[0], self.number // 2), :]
        query = torch.cat((query_p, query_n), dim=0)
        # Shuffle query
        shuffle_idx = torch.randperm(self.number)
        query = query[shuffle_idx]
        query_label = query_label[shuffle_idx]

        return query, query_label, class_sample

    def __len__(self):
        '''

        Returns: Length of the Dataset

        '''
        return self.dataset.shape[0]