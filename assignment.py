import os
import argparse
import numpy as np
from preprocessing import DataProcessor
import tensorflow as tf
from typing import Optional
from types import SimpleNamespace
from model import CoLSTM
from model import LSTM
from model import CNN

def parse_args(args=None):
    """ 
    Perform command-line argument parsing (otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Let's train some neural nets!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task',           choices=['train', 'test', 'both'],  help='Task to run', default='both')
    parser.add_argument('--epochs',         type=int,   default=5,      help='Number of epochs used in training.')
    parser.add_argument('--lr',             type=float, default=1e-3,   help='Model\'s learning rate')
    parser.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Model\'s optimizer')
    parser.add_argument('--batch_size',     type=int,   default=64,    help='Model\'s batch size.')
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--data_source',    choices=['movies', 'airlines', 'elections', 'anime'],   default='movies',    help='Source of data for model')
    parser.add_argument('--model',    choices=['colstm', 'lstm', 'cnn'],   default='colstm',    help='model architecture to run')
    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

def main(args):

    ##############################################################################
    ## Data Loading
    processor = DataProcessor()
    data_dict = processor.load_data(args.data_source) # must run preprocessing.py first for data loading to be successful 

    train_text  = np.array(data_dict['train_reviews']) 
    test_text   = np.array(data_dict['test_reviews'])
    train_labels = data_dict['train_labels']
    test_labels  = data_dict['test_labels']

    vocabulary        = data_dict['vocabulary']

    model = None
    if args.model == "colstm":
        model = CoLSTM(vocab_size=(len(vocabulary) + 1))
    elif args.model == "lstm":
        model = LSTM(vocab_size=(len(vocabulary) + 1))
    elif args.model == "cnn":
        model = CNN(vocab_size=(len(vocabulary) + 1))

    ##############################################################################
    ## Training Task
    if args.task in ('train', 'both'):
        train_model(model, train_text, train_labels, args)
                
    ##############################################################################
    ## Testing Task
    if args.task in ('test', 'both'):
        test_model(model, test_text, test_labels, args)

    ##############################################################################

def train_model(model, reviews, labels, args):
    '''Trains model and returns model statistics'''
    try:
        for epoch in range(args.epochs):
            total_loss, total_accuracy = model.train(reviews, labels, batch_size=args.batch_size)
            print(f"Train Epoch: {epoch} \tLoss: {tf.math.reduce_mean(total_loss):.6f} \tAccuracy: {tf.math.reduce_mean(total_accuracy):.6f}")
    except KeyboardInterrupt as e:
        if epoch > 0:
            print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
        else: 
            raise e

def test_model(model, reviews, labels, args):
    '''Tests model and returns model statistics'''
    total_loss, total_accuracy = model.test(reviews, labels)
    print(f"Test \tLoss: {tf.math.reduce_mean(total_loss):.6f} \tAccuracy: {tf.math.reduce_mean(total_accuracy):.6f}")

if __name__ == '__main__':
    main(parse_args())
