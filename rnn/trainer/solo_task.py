import argparse

import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        help='Path to training data',
                        required=True
                        )
    parser.add_argument('--output',
                        help='Path to output folder',
                        required=True
                       )
    parser.add_argument('--batch_size',
                        help='Batch size',
                        type=int,
                        default=256
                      )
    parser.add_argument('--epochs',
                        help='Epochs',
                        type=int,
                        default=100
                        )

    args = parser.parse_args()

    model.DATA_PATH = args.data
    model.OUTPUT_PATH = args.output
    model.BATCH_SIZE = args.batch_size
    model.EPOCHS = args.epochs

    model.train()

