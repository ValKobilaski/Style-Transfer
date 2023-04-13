# Program to load images and their corresponding segmentation masks
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Helper class to load image data from file
class ImageDataset(Dataset):
    def __init__(self, image_root, mask_root):
        """
        Dataset to load image and mask pairs
        :param image_root: Root directory containing images
        :param mask_root: Root directory containing masks
        """
        self.img_transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
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
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        mask = Image.open(self.image_pairs[idx][1])
        # Apply transform
        image, mask = self.img_transform(image), self.mask_transform(mask)
        return image, mask

def get_loader(image_root, mask_root, batch_size, shuffle=True):
    dataset = ImageDataset(image_root, mask_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

if __name__ == '__main__':

    image_root = 'data/train'
    mask_root = 'data/annotations/stuff_train2017_pixelmaps'

    # Create dataset
    dataset = ImageDataset(image_root, mask_root)

    # Create dataloader
    dataloader = get_loader(image_root, mask_root, batch_size=4)

    # Test dataloader
    for image, mask in dataloader:
        print(image.shape)
        print(mask.shape)
        break