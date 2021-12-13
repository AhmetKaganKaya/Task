import torch
from utils import calculate_accuracy
from tqdm import tqdm
from torch.utils.data import DataLoader
import os


def train(args, train_dataset, test_dataset, model, optimizer, loss, logger, device,):
    train_losses = []
    test_acc = []
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size)
    best_accuracy = None
    iteration_train = iter(train_loader)
    iteration_test = iter(test_loader)

    for epoch in range(args.num_iter):
        logger.info("Epoch {}".format(epoch + 1))
        model.train()
        epoch_loss = 0
        train_bar = tqdm(range(args.train_iter))
        for _ in train_bar:
            (query, target, sample) = next(iteration_train)
            train_bar.set_description(
                "Epoch Loss: {:.2f}".format(epoch_loss))
            query = query.to(device)
            target = target.to(device)
            sample = sample.to(device)

            optimizer.zero_grad()
            output = model(sample, query)
            batch_loss = loss(output.squeeze(-1), target)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss
        logger.info("Epoch Loss: {:.2f}".format(epoch_loss))
        train_losses.append(epoch_loss)

        accuracy = test(args, model, iteration_test, device, logger)
        if best_accuracy is None or best_accuracy > accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(args.save, "checkpoint", args.module_type + '_best_model.pth'))
        test_acc.append(accuracy)

    return train_losses, test_acc

def test(args, model, test_loader, device, logger):
    correct = 0
    model.eval()
    test_bar = tqdm(range(args.test_iter))
    for _ in test_bar:
        (query, target, sample) = next(test_loader)
        test_bar.set_description(
            "Test Accuracy: {:.2f}%".format((100 * correct) / (args.query_size * args.batch_size * 50)))
        query = query.to(device)
        target = target.to(device)
        sample = sample.to(device)

        output = model(sample, query)
        output = torch.where(output < 0.5, 0., 1.)
        correct += calculate_accuracy(output.squeeze(-1), target)
    accuracy = (100 * correct) / (args.query_size * args.batch_size * args.test_iter)
    logger.info("Test Accuracy: {:.2f}%".format((100 * correct) / (args.query_size * args.batch_size * 50)))

    return accuracy








