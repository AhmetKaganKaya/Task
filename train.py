import torch
from utils import calculate_accuracy
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import optuna




def train(trial, args, train_dataset, test_dataset, model, loss, logger, device,):
    '''
    This method takes dataset, model, optimizer and loss metric and train model to learn task
    Args:
        args: Argument List
        train_dataset: DataLoader of the Train Dataset
        test_dataset: DataLoader of the Test Dataset
        model: Model that we want to train: MLP or CNN
        optimizer: Optimizer (Adam)
        loss: Loss metric (Binary Cross Entropy Loss)
        logger: logger file
        device: GPU
    Returns:
        train_losses: All Train Losses epoch by epoch
        test_acc: All Test accuracy epoch by epoch
    '''
    try:
        train_losses = []
        test_acc = []
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size)
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size)
        best_accuracy = None
        args.lr = trial.suggest_float("lr", 1e-5, 0.1, log=True)
        print("Suggested LR: {}".format(args.lr))
        optimizer = optim.Adam(params=model.parameters(), lr= args.lr)
        for epoch in range(args.num_iter):
            logger.info("Epoch {}".format(epoch + 1))
            model.train()
            epoch_loss = 0
            train_bar = tqdm(enumerate(train_loader))
            for idx, (query, target, sample) in train_bar:
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
                if idx == args.train_iter:
                    break

            logger.info("Epoch Loss: {:.2f}".format(epoch_loss))
            train_losses.append(epoch_loss)

            accuracy = test(args, model, test_loader, device, logger)
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if best_accuracy is None or best_accuracy > accuracy:
                best_accuracy = accuracy
                best_model = model.state_dict()
                torch.save(best_model, os.path.join(args.save, "checkpoint", args.module_type + '_best_model' + str(args.device) + '.pth'))
            test_acc.append(accuracy)
    except KeyboardInterrupt as ke:
        logger.info('======= Summary =======')
        logger.info('Best Accuracy: {}'.format(best_accuracy))
        torch.save(best_model, os.path.join(args.save, "checkpoint", args.module_type + '_best_model' + str(args.device) + '.pth'))

    return best_accuracy


def test(args, model, test_loader, device, logger):
    '''
    This model takes trained model and test dataset. Measure accuracy and returns again train function.
    Args:
        args: Argument List
        model: Model that we want to train: MLP or CNN
        test_loader: DataLoader of the Test Dataset
        device: GPU
        logger: logger file
    Returns:
        accuracy: Accuracy value
    '''
    correct = 0
    model.eval()
    test_bar = tqdm(enumerate(test_loader))
    for idx, (query, target, sample) in test_bar:
        test_bar.set_description(
            "Test Accuracy: {:.2f}%".format((100 * correct) / (args.query_size * args.batch_size * args.test_iter)))
        query = query.to(device)
        target = target.to(device)
        sample = sample.to(device)

        output = model(sample, query)
        output = torch.where(output < 0.5, 0., 1.)
        correct += calculate_accuracy(output.squeeze(-1), target)
        if idx == args.test_iter:
            break
    accuracy = (100 * correct) / (args.query_size * args.batch_size * args.test_iter)
    logger.info("Test Accuracy: {:.2f}%".format((100 * correct) / (args.query_size * args.batch_size * args.test_iter)))

    return accuracy








