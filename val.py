import torch
from torch.utils.data import DataLoader

import argparse

from resnet34 import ResNet34
from dataset import Animals10Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', dest='batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--load', '-f', type=str, required=True, help='path to model')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}.')

    dataset = Animals10Dataset(set_type='val')
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = ResNet34(10).to(device)
    state_dict = torch.load(args.load)
    model.load_state_dict(state_dict['model'])
    print(f'Loaded model from {args.load}!')
    model.eval()

    total = 0
    correct = 0

    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)
        pred = model(img)
        pred = torch.argmax(pred, dim=1)
        total += pred.size(0)
        correct += torch.sum(torch.eq(pred, label)).item()

    accuracy = correct / total
    print(f'Validation finished! Accuracy: {accuracy:.2%}')


if __name__ == '__main__':
    main()
