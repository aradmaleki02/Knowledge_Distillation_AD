from argparse import ArgumentParser
from utils.utils import get_config
from dataloader import load_data, load_localization_data
from test_functions import detection_test, localization_test
from models.network import get_networks
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
from PIL import Image
from glob import glob

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.dataset import Subset

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--root', type=str, default='Br35H', help='training dataset')


class Brain_MRI(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)

def import_loaders(batch_size, root, first_dataset=True):
    train_normal_path = glob(f'./{root}/dataset/train/normal/*')
    train_label = [0] * len(train_normal_path)
    test_normal_path = glob(f'./{root}/dataset/test/normal/*')
    test_anomaly_path = glob(f'./{root}/dataset/test/anomaly/*')

    print(f'./{root}/dataset/train/normal/')
    # print(train_normal_path, test_normal_path, test_anomaly_path)
    print('len(train normal, test normal, test anomaly): ', len(train_normal_path), len(test_normal_path), len(test_anomaly_path))

    test_path = test_normal_path + test_anomaly_path
    test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    whole_test_path = train_normal_path + test_path
    whole_test_label = [0] * len(train_normal_path) + test_label

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = Brain_MRI(image_path=train_normal_path, labels=train_label, transform=transform)
    test_set = Brain_MRI(image_path=test_path, labels=test_label, transform=transform)
    whole_test_set = Brain_MRI(image_path=whole_test_path, labels=whole_test_label, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=batch_size)
    whole_test_loader = torch.utils.data.DataLoader(whole_test_set, shuffle=False, batch_size=batch_size)

    if first_dataset:
        return train_loader, test_loader
    return train_loader, whole_test_loader

def main():
    args = parser.parse_args()
    config = get_config(args.config)
    root = args.root
    vgg, model = get_networks(config, load_checkpoint=True)

    # Localization test
    if config['localization_test']:
        test_dataloader, ground_truth = load_localization_data(config)
        roc_auc = localization_test(model=model, vgg=vgg, test_dataloader=test_dataloader, ground_truth=ground_truth,
                                    config=config)

    # Detection test
    else:
        _, test_dataloader = import_loaders(config['batch_size'], root, first_dataset=False)
        roc_auc = detection_test(model=model, vgg=vgg, test_dataloader=test_dataloader, config=config)
    last_checkpoint = config['last_checkpoint']
    print("RocAUC after {} epoch:".format(last_checkpoint), roc_auc)


if __name__ == '__main__':
    main()
