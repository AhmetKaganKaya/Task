import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader



def train(args, train_dataset, test_dataset, model, optimizer, loss, device, batch_size):
    train_losses = []
    test_acc = []
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, drop_last=True, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, drop_last=True, batch_size=batch_size)
    for epoch in range(args.num_iter):
        print("############Epoch{}############".format(epoch + 1))
        model.train()
        epoch_loss = 0
        for idx, (query, target, sample) in enumerate(train_loader):
            query = query.to(device)
            target = target.to(device)
            sample = sample.to(device)

            optimizer.zero_grad()
            output = model(sample, query)
            batch_loss = loss(output.squeeze(2), target)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss
            if idx == 100:
                break
        train_losses.append(epoch_loss)
        print("Train Loss: {:.2f}".format(epoch_loss))

        correct = 0
        model.eval()
        for idx, (query, target, sample) in enumerate(test_loader):
            query = query.to(device)
            target = target.to(device)
            sample = sample.to(device)

            output = model(sample, query)
            output = torch.where(output < 0.5, 0., 1.)
            correct += (output.squeeze(2)).eq(target).cpu().sum()
            if idx == 50:
                break
        accuracy = (100 * correct)/(args.query_size * batch_size *  50)
        test_acc.append(accuracy)
        print("Test Accuracy: {:.2f}%".format(accuracy))

    return train_losses, test_acc








