import torch
import random
from tqdm import tqdm


def train(args, dataset, model, optimizer, loss, device, train_set, test_set):
    for epoch in range(args.num_iter):
        model.train()
        epoch_loss = 0
        dataset.set_training_samples()
        number = random.choice(train_set)
        batches = dataset.get_batch(number)
        for batch in batches:
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
        print(f"EpochTrain Loss: {epoch_loss}")

        correct = 0
        model.eval()
        for i in tqdm(range(args.test_iter)):
            dataset.set_training_samples()
            num = random.choice(test_set)
            batches = dataset.get_batch(num)
            batch = batches[random.randint(0, len(batches) - 1)]
            query, label, sample = batch
            query = query.to(device)
            label = label.to(device)
            sample = sample.to(device)

            output = model.forward(sample.unsqueeze(0), query)
            output = torch.where(output < 0.5, 0., 1.)
            correct += output.eq(label).cpu().sum()
        print(f"Test Accuracy: {(100 * correct)/(args.query_size * args.test_iter)}")








