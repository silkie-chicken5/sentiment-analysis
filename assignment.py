import os
import argparse
import numpy as np
import pickle
from preprocessing_movies import load_data 
import tensorflow as tf
from typing import Optional
from types import SimpleNamespace
from model import CoLSTM

# from model import ImageCaptionModel, accuracy_function, loss_function
# from decoder import TransformerDecoder, RNNDecoder
# import transformer


def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Let's train some neural nets!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--type',           required=True,              choices=['rnn', 'transformer'],     help='Type of model to train')
    parser.add_argument('--task',           choices=['train', 'test', 'both'],  help='Task to run', default='both')
    parser.add_argument('--data',           help='File path to the assignment data file.', default='./data/imdb_test.csv')
    parser.add_argument('--epochs',         type=int,   default=5,      help='Number of epochs used in training.')
    parser.add_argument('--lr',             type=float, default=1e-3,   help='Model\'s learning rate')
    parser.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Model\'s optimizer')
    parser.add_argument('--batch_size',     type=int,   default=30,    help='Model\'s batch size.')
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    # parser.add_argument('--window_size',    type=int,   default=20,     help='Window size of text entries.')
    # parser.add_argument('--chkpt_path',     default='',                 help='where the model checkpoint is')
    # parser.add_argument('--check_valid',    default=True,               action="store_true",  help='if training, also print validation after each epoch')
    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.


def main(args):

    ##############################################################################
    ## Data Loading
    # with open(args.data, 'rb') as data_file: DOES NOT WORK WITH CSVs
    #     data_dict = pickle.load(data_file)
    print("in main")
    
    data_dict = load_data() # will need to replace with argument to decide which data to load

    train_text  = np.array(data_dict['train_reviews']) 
    test_text   = np.array(data_dict['test_reviews'])
    # print("train text: ")
    # print(train_text)
    # print("test text: ")
    # print(test_text)
    # print(data_dict['train_labels'].shape)
    # print(data_dict['test_labels'].shape)
    train_labels = data_dict['train_labels'] # shape: (40000,)
    test_labels  = data_dict['test_labels'] # shape: (10000,)

    vocabulary        = data_dict['vocabulary']

    # Vestigial Code
    # train_images    = img_prep(data_dict['train_images'])
    # test_images     = img_prep(data_dict['test_images'])
    # idx2word        = data_dict['idx2word']
    # feat_prep = lambda x: np.repeat(np.array(x).reshape(-1, 2048), 5, axis=0)


    model = CoLSTM(vocab_size=(len(vocabulary) + 1))

    ##############################################################################
    ## Training Task
    if args.task in ('train', 'both'):
        ##############################################################################
        
        # compile_model(model, args)
        # train_model(
        #     model, train_text, train_labels, word2idx['<pad>'], args, 
        #     valid = (test_text, test_labels)
        # )
        # if args.chkpt_path: 
        #     ## Save model to run testing task afterwards
        #     save_model(model, args)
        # for epoch_id in range(args.num_epochs):
        train_model(model, train_text, train_labels, args)
            # print(f"Train Epoch: {epoch_id} \tLoss: {total_loss / len(train_text):.6f}")
                
    ##############################################################################
    ## Testing Task
    if args.task in ('test', 'both'):
        # if args.task != 'both': 
            ## Load model for testing. Note that architecture needs to be consistent
            # model = load_model(args)

        # if not (args.task == 'both' and args.check_valid):
        #    test_model(model, test_text, test_labels, word2idx['<pad>'], args)
        total_loss, total_accuracy = test_model(model, test_text, test_labels, args)
        # print("total loss manual calculation: ", tf.math.reduce_mean(total_loss))
        print(f"Test \tLoss: {tf.math.reduce_mean(total_loss):.6f} \tAccuracy: {tf.math.reduce_mean(total_accuracy):.6f}")
        # print(f"Test \tLoss: {np.mean(total_loss / len(train_text)):.6f}")

    ##############################################################################

##############################################################################
## UTILITY METHODS

# def save_model(model, args):
#     '''Loads model based on arguments'''
#     os.makedirs(f"{args.chkpt_path}", exist_ok=True)

#     tf.keras.models.save_model(model, args.chkpt_path)
#     print(f"Model saved to {args.chkpt_path}")


# def load_model(args):
#     '''Loads model by reference based on arguments. Also returns said model'''
#     model = tf.keras.models.load_model(
#         args.chkpt_path,
#         custom_objects=dict(
#             AttentionHead           = transformer.AttentionHead,
#             AttentionMatrix         = transformer.AttentionMatrix,
#             MultiHeadedAttention    = transformer.MultiHeadedAttention,
#             TransformerBlock        = transformer.TransformerBlock,
#             PositionalEncoding      = transformer.PositionalEncoding,
#             TransformerDecoder      = TransformerDecoder,
#             RNNDecoder              = RNNDecoder,
#             ImageCaptionModel       = ImageCaptionModel
#         ),
#     )
    
    ## Saving is very nuanced. Might need to set the custom components correctly.
    ## Functools.partial is a function wrapper that auto-fills a selection of arguments. 
    ## so in other words, the first argument of ImageCaptionModel.test is model (for self)
    # from functools import partial
    # model.test    = partial(ImageCaptionModel.test,    model)
    # model.train   = partial(ImageCaptionModel.train,   model)
    # model.compile = partial(ImageCaptionModel.compile, model)
    # compile_model(model, args)
    # print(f"Model loaded from '{args.chkpt_path}'")
    # return model


# def compile_model(model, args):
#     '''Compiles model by reference based on arguments'''
#     optimizer = tf.keras.optimizers.get(args.optimizer).__class__(learning_rate = args.lr)
#     model.compile(
#         optimizer   = optimizer,
#         loss        = loss_function,
#         metrics     = [accuracy_function]
#     )


def train_model(model, reviews, labels, args):
    '''Trains model and returns model statistics'''
    # stats = []
    print("in train_model")
    try:
        for epoch in range(args.epochs):
            # stats += [model.train(reviews, labels, batch_size=args.batch_size)]
            total_loss, total_accuracy = model.train(reviews, labels, batch_size=args.batch_size)
            # print("total loss manual calculation: ", tf.math.reduce_mean(total_loss))
            # print("len of dataset is: ", len(reviews))
            # print("test val: ", tf.math.reduce_mean(total_loss / len(reviews)))
            print(f"Train Epoch: {epoch} \tLoss: {tf.math.reduce_mean(total_loss):.6f} \tAccuracy: {tf.math.reduce_mean(total_accuracy):.6f}")
            # if args.check_valid:
            #     model.test(valid[0], , batch_size=args.batch_size)
    except KeyboardInterrupt as e:
        if epoch > 0:
            print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
        else: 
            raise e
        
    # return stats


def test_model(model, reviews, labels, args):
    '''Tests model and returns model statistics'''
    print("in test_model")
    # perplexity, accuracy = model.test(reviews, labels, batch_size=args.batch_size)
    total_loss, total_accuracy = model.test(reviews, labels, batch_size=args.batch_size)
    # return perplexity, accuracy
    return total_loss, total_accuracy


## END UTILITY METHODS
##############################################################################

## 
if __name__ == '__main__':
    main(parse_args())
