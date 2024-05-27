import torch

import argparse


def main():
    parser = argparse.ArgumentParser(description='Train ResNet for classification')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='How many epochs to train')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load a model to continue training')
    args = parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()
