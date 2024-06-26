import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse

from resnet34 import ResNet34
from dataset import Animals10Dataset


def main():
    parser = argparse.ArgumentParser(description='Train ResNet for classification')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='How many epochs to train')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num-workers', '-w', dest='num_workers', type=int, default=8,
                        help='Number of workers for DataLoader')
    parser.add_argument('--learning-rate', '-l', dest='lr', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False,
                        help='Load a model to continue training')
    parser.add_argument('--n-to-val', '-v', dest='n_to_val', type=int, default=50,
                        help='How many iterations per validation')
    parser.add_argument('--n-to-save', '-s', dest='n_to_save', type=int, default=400,
                        help='How many iterations per saving')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}.')

    train_dataset = Animals10Dataset(set_type='train')
    val_dataset = Animals10Dataset(set_type='val')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=0)

    model = ResNet34(10)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    iterations = 0

    if args.load:
        state_dict = torch.load(args.load)
        iterations = state_dict['iterations']
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        print(f'Loaded model from {args.load}!')

    for epoch in range(args.epochs):
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            pred = model(img)
            loss = criterion(pred, label)
            loss.backward()

            optimizer.step()

            print(f'Iteration {iterations} finished with loss: {loss.item():.4f}.')
            iterations += 1

            if iterations % args.n_to_val == 0:
                with torch.no_grad():
                    total = 0
                    total_correct = 0
                    for img_val, label_val in val_loader:
                        img_val = img_val.to(device)
                        label_val = label_val.to(device)
                        pred = torch.argmax(model(img_val), dim=1)
                        total_correct += torch.sum(torch.eq(pred, label_val))
                        total += pred.size(0)
                    accuracy = total_correct / total
                    print(f'Validation finished with accuracy: {accuracy:.2%}.')

            if iterations % args.n_to_save == 0:
                save_path = f'./checkpoints/animals-10-resnet34-{iterations}.pth'
                state_dict = {
                    'iterations': iterations,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state_dict, save_path)
                print(f'Model saved at {save_path}!')


if __name__ == '__main__':
    main()
