import logging
import matplotlib.pyplot as plt

def setLogger(logFilePath):
    logHandler = [logging.FileHandler(logFilePath), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=logHandler)
    logger = logging.getLogger()

    return logger

def calculate_accuracy(output, target):
    return output.eq(target).cpu().sum()

def visualize_results(args, dict):
    length = len(dict)
    for i in range(length):
        plt.subplot(1, length, i+1)
        plt.plot(dict[list(dict.keys())[i]])
        plt.title(list(dict.keys())[i])
    plt.savefig("output/" + args.module_type + "_results.png")