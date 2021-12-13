import logging

def setLogger(logFilePath):
    logHandler = [logging.FileHandler(logFilePath), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=logHandler)
    logger = logging.getLogger()

    return logger

def calculate_accuracy(output, target):
    return output.eq(target).cpu().sum()