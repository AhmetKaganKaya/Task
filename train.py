import torch
import random
from tqdm import tqdm


def train(args, dataset, model, optimizer, loss, device, train_set, test_set):
    train_losses = []
    test_acc = []
    for epoch in range(args.num_iter):
        model.train()
        epoch_loss = 0
        dataset.set_training_samples()
        number = random.choice(train_set)
        print(f"Epoch {epoch + 1}! ---> sampled class {number}")
        batches = dataset.get_train_batches(number)
        for batch in tqdm(batches):
            query, label, sample = batch
            query = query.to(device)
            label = label.to(device)
            sample = sample.to(device)

            optimizer.zero_grad()
            output = model.forward(sample.unsqueeze(0), query)
            batch_loss = loss(output, label)
            epoch_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
        train_losses.append(epoch_loss)
        print("Train Loss: {:.2f}".format(epoch_loss))

        correct = 0
        model.eval()
        for i in tqdm(range(args.test_iter)):
            dataset.set_training_samples()
            num = random.choice(test_set)
            batch = dataset.get_test_batch(num)
            query, label, sample = batch
            query = query.to(device)
            label = label.to(device)
            sample = sample.to(device)

            output = model.forward(sample.unsqueeze(0), query)
            output = torch.where(output < 0.5, 0., 1.)
            correct += output.eq(label).cpu().sum()
        accuracy = (100 * correct)/(args.query_size * args.test_iter)
        test_acc.append(accuracy)
        print("Test Accuracy: {:.2f}%".format(accuracy))

    return train_losses, test_acc








