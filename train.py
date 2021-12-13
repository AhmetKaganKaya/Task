import torch
from utils import calculate_accuracy
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import optuna


def train(trial, args, train_dataset, test_dataset, model, loss, logger, device,):
    try:
        train_losses = []
        test_acc = []
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size)
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size)
        best_accuracy = None
        iteration_train = iter(train_loader)
        iteration_test = iter(test_loader)
        args.lr = trial.suggest_float("lr", 1e-5, 0.1, log=True)
        print("Suggested LR: {}".format(args.lr))
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay= 1e-4)

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
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if best_accuracy is None or best_accuracy > accuracy:
                best_accuracy = accuracy
                best_model = model.state_dict()
                torch.save(best_model, os.path.join(args.save, "checkpoint", args.module_type + '_best_model.pth'))
            test_acc.append(accuracy)
    except KeyboardInterrupt as ke:
        logger.info('======= Summary =======')
        logger.info('Best Accuracy: {}'.format(best_accuracy))
        torch.save(best_model, os.path.join(args.save, "checkpoint", args.module_type + '_best_model.pth'))

    return accuracy

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








