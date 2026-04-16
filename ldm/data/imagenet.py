"""
ImageNet dataset loaders
"""

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageNetBase(Dataset):
    """Base ImageNet dataset."""
    def __init__(self, data_root, size=256, split='train'):
        self.data_root = data_root
        self.size = size
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # to [-1, 1]
        ])

        # Load image paths
        self.image_paths = []
        split_dir = os.path.join(data_root, split)
        if os.path.exists(split_dir):
            for class_dir in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_dir)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            self.image_paths.append(os.path.join(class_path, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {'image': image}


class ImageNetTrain(ImageNetBase):
    def __init__(self, data_root, size=256):
        super().__init__(data_root, size, split='train')


class ImageNetValidation(ImageNetBase):
    def __init__(self, data_root, size=256):
        super().__init__(data_root, size, split='val')
