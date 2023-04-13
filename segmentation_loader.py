# Program to load images and their corresponding segmentation masks
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt

# Helper class to load image data from file
class ImageDataset(Dataset):
    def __init__(self, image_root, mask_root):
        """
        Dataset to load image and mask pairs
        :param image_root: Root directory containing images
        :param mask_root: Root directory containing masks
        :param transform: Transform to apply to image and mask
        :param class_encoding: One-hot encoding for each class
        """
        self.img_transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.NEAREST_EXACT),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.NEAREST_EXACT),
            transforms.CenterCrop(256),
            transforms.PILToTensor(),
        ])
        self.image_pairs = []
        # Get all image files in subdirectories of image_root
        for root, _, files in os.walk(image_root):
            for file in files:
                if file.endswith(".jpg"):
                    # Get corresponding mask file
                    mask_file = os.path.join(mask_root, file)
                    # Change extension to png
                    mask_file = os.path.splitext(mask_file)[0] + ".png"
                    # Add to list of image pairs
                    self.image_pairs.append((os.path.join(root, file), mask_file))


    def __len__(self):
        return len(self.image_pairs)
    def __getitem__(self, idx):
        """
        Returns tuple containing image and one-hot encoded mask
        """
        # Load image and mask
        image = Image.open(self.image_pairs[idx][0])
        mask = Image.open(self.image_pairs[idx][1])
        # Apply transform
        image, mask = self.img_transform(image), self.mask_transform(mask)
        return image, mask

image_root = 'data/train'
mask_root = 'data/annotations/stuff_train2017_pixelmaps'

# Create dataset
dataset = ImageDataset(image_root, mask_root)

# Get first image and mask
image, mask = dataset[0]

# Plot image and mask
import matplotlib.pyplot as plt
plt.imshow(image.permute(1, 2, 0))
plt.show()
plt.imshow(mask.permute(1, 2, 0))
plt.show()