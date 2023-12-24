from parser_custom import parse_arguments
import os
import logging
from classification_training import Training
from load_data import load_data

def main(opt):

    training_loader, validation_loader = load_data(opt["data_path"])
    experiment = Training(opt)
    loss = experiment.train(training_loader)
    validation_loss = experiment.validate(validation_loader)



if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)