import torch
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()
                                # ,transforms.RandomCrop(32, padding=4)
                                # ,transforms.RandomHorizontalFlip()
                                ,transforms.Normalize((0.4914, 0.4822, 0.4465),      #CIFAR-10 Mean
                                                     (0.2471, 0.2435, 0.2616)) #CIFAR-10 STD
                                                     ])    


def get_datasets(download_path = './data'):
    train_ds = CIFAR100(root = download_path, train = True, transform = transform, download = True)
    test_ds = CIFAR100(root = download_path, train = False, transform = transform, download = True)
    return train_ds, test_ds


def dataset_to_dataloader(dataset, generator, batch_size = 64, shuffle = True):
    return DataLoader(dataset, batch_size= batch_size, shuffle = shuffle, num_workers = 2, generator=generator)

def test_dataloader(dataloader):
    """
    For testing purposes: displays a grid of images from
    a specified dataloader.
    """
    for img,lb in dataloader:
        fig, ax = plt.subplots(figsize=(16,8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(img.cpu(), nrow=16).permute(1,2,0))
        break