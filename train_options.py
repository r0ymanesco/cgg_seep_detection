import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_batch_size', required=True, type=int,
                    help='Training batch size.')
parser.add_argument('--eval_batch_size', required=True, type=int,
                    help='Evaluation batch size. Same as validation batch size.')
parser.add_argument('--epochs', required=True, type=int,
                    help='Maximum number of epochs for training')
parser.add_argument('--lr', required=True, type=float,
                    help='Learning rate.')