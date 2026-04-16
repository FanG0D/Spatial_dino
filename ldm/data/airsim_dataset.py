"""
AirSim drone simulation dataset loader
Loads images from episode-based directory structure
"""

import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class AirSimDataset(Dataset):
    """
    AirSim drone simulation dataset.
    Data structure: episode_folder/*.jpg
    Example: 302OLP89E75X7MFPRR58QG7CIDVACC_processed/0.jpg
    """

    def __init__(self, data_root, size=256, split='train', train_ratio=0.9):
        """
        Args:
            data_root: Path to airvln_16 directory
            size: Image size for resizing
            split: 'train' or 'val'
            train_ratio: Ratio of episodes for training (default 0.9)
        """
        self.data_root = data_root
        self.size = size
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # to [-1, 1]
        ])

        # Get all episode directories
        episode_dirs = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and '_processed' in d
        ])

        # Split episodes into train/val
        n_train = int(len(episode_dirs) * train_ratio)
        if split == 'train':
            selected_episodes = episode_dirs[:n_train]
        else:
            selected_episodes = episode_dirs[n_train:]

        # Collect all image paths from selected episodes
        self.image_paths = []
        for ep_dir in selected_episodes:
            ep_path = os.path.join(data_root, ep_dir)
            # Get all jpg images in this episode
            images = sorted(glob.glob(os.path.join(ep_path, '*.jpg')))
            self.image_paths.extend(images)

        print(f"AirSim {split} dataset: {len(selected_episodes)} episodes, "
              f"{len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {'image': image}


class AirSimTrain(AirSimDataset):
    """AirSim training dataset"""

    def __init__(self, data_root, size=256):
        super().__init__(data_root, size, split='train')


class AirSimValidation(AirSimDataset):
    """AirSim validation dataset"""

    def __init__(self, data_root, size=256):
        super().__init__(data_root, size, split='val')
