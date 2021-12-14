import logging
import matplotlib.pyplot as plt

def setLogger(logFilePath):
    '''
    This method enable user to create logger object and write training details
    Args:
        logFilePath: path of the .txt file

    Returns:
        logger: logger object that enable user to write .txt file
    '''
    logHandler = [logging.FileHandler(logFilePath), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=logHandler)
    logger = logging.getLogger()

    return logger

def calculate_accuracy(output, target):
    '''

    Args:
        output: labels that come from model
        target: labels that come from dataset

    Returns:
        accuracy: This value shows haw many value are labeled correctly
    '''
    accuracy = output.eq(target).cpu().sum()
    return accuracy

def visualize_result(args,dict):
    '''

    Args:
        args: Argument list
        dict: Dictionary's keys are plot title and value of these keys are the any list (Training Loss or Test Accuracy)

    Returns:

    '''
    length = len(dict)
    for i in range(length):
        plt.subplot(1,length, i+1)
        plt.plot(dict[list(dict.keys())[i]])
        plt.title(list(dict.keys())[i])
    plt.savefig("output/" + args.module_type + "_result.png")

