import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

import json
import os


class Animals10Dataset(Dataset):
    def __init__(self, set_type='train', index_root='./data', dataset_root='./data/animals10'):
        self.classes = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
        self.index_root = index_root
        self.dataset_root = dataset_root

        index_path = os.path.join(index_root, f'index-{set_type}.json')
        with open(index_path, 'r') as f:
            index_json = f.read()
        self.index_data = json.loads(index_json)

    def _class2label(self, name):
        return self.classes.index(name)

    def __getitem__(self, index):
        sample_data = self.index_data[index]
        file_path = sample_data['file']
        label = self._class2label(sample_data['class'])
        img = read_image(os.path.join(self.dataset_root, file_path))
        # 归一化 mean = 127.5, std = 127.5
        img = (img - 127.5) / 127.5
        return img, label

    def __len__(self):
        return len(self.index_data)
