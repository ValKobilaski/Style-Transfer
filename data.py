from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader

def get_dataset(path, img_size):

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),#COCO Mean
                            (0.229, 0.224, 0.225)) #COCO STD                  
    ])
    return datasets.ImageFolder(path, transform)

def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size= batch_size, shuffle = shuffle, num_workers = 2)