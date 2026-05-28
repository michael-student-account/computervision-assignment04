# A HELPER FILE FOR CLASSES AND FUNCTIONS THAT IMPORT AND TRANSFORM THE DATASET
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from collections import defaultdict
import random

# CUSTOM DATASET CLASS - FOR TRAINING
class VeRiDataset_Train(Dataset):
    """
    A custom dataset class for the VeRi dataset that translates the raw 
    datset folder into a useable format for PyTorch dataloaders.
    Adapted from a shell structure suggested by ChatGPT.
    """
    def __init__(self, image_dir, transform):
        # Initialize directory and transformations
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(list(self.image_dir.glob('*.jpg')))
        self.transform = transform

        # Vehicle IDs
        self.vehicle_ids = [
            self._parse_vehicle_id(path) for path in self.image_paths
        ]

        # Build dictionary of matching vehicle ids for sampling positive anchors
        self.id_to_indices = defaultdict(list)

        for idx, vehicle_id in enumerate(self.vehicle_ids):
            self.id_to_indices[vehicle_id].append(idx)
        
        # Inspect and warn if any vehicle IDs have only one image
        # From ChatGPT
        single_image_ids = [
            vehicle_id for vehicle_id, indices in self.id_to_indices.items()
            if len(indices) < 2
        ]

        if len(single_image_ids) > 0:
            print(f"Warning: {len(single_image_ids)} vehicle IDs have only one image.")
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]  # Get image path for the given idx
        image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
        vehicle_id = self.vehicle_ids[idx]  # Get vehicle ID for the given idx
        
        # Randomly choose a positive anchor
        unique_positive = False

        while not unique_positive:
            pos_anchor_idx = random.choice(self.id_to_indices[vehicle_id])
            if pos_anchor_idx != idx:
                unique_positive = True

        pos_image_path = self.image_paths[pos_anchor_idx]
        pos_image = Image.open(pos_image_path).convert('RGB')
        pos_id = self.vehicle_ids[pos_anchor_idx]
        
        # Randomly choose a negative anchor
        unique_negative = False

        while not unique_negative:
            neg_anchor_idx = random.randint(0, len(self.image_paths) - 1)
            neg_id = self.vehicle_ids[neg_anchor_idx]

            if neg_id != vehicle_id:
                unique_negative = True

        neg_image_path = self.image_paths[neg_anchor_idx]
        neg_image = Image.open(neg_image_path).convert('RGB')

        # Transforms     
        image = self.transform(image)
        pos_image = self.transform(pos_image)
        neg_image = self.transform(neg_image)
        
        return {
            "image": image,
            "vehicle_id": vehicle_id,
            "pos_anchor": pos_image,
            "pos_id": pos_id,
            "neg_anchor": neg_image,
            "neg_id": neg_id
        }

    @staticmethod
    def _parse_vehicle_id(path):
        "First part of the file path gives the vehicle ID"
        vehicle_id = path.stem.split('_')[0]
        return int(vehicle_id)

# CUSTOM DATASET CLASS - FOR EVALUATION
class VeRiDataset_Eval(Dataset):
    """
    A custom dataset class for the VeRi dataset that translates the raw 
    datset folder into a useable format for PyTorch dataloaders.
    Adapted from a shell structure suggested by ChatGPT.
    """
    def __init__(self, image_dir, transform):
        # Initialize directory and transformations
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(list(self.image_dir.glob('*.jpg')))
        self.transform = transform

        # Vehicle IDs
        self.vehicle_ids = [
            self._parse_vehicle_id(path) for path in self.image_paths
        ]
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]  # Get image path for the given idx
        image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
        vehicle_id = self.vehicle_ids[idx]  # Get vehicle ID for the given idx

        if self.transform:
            image = self.transform(image)  # Apply transformations if provided
        
        return {
            "image": image,
            "vehicle_id": vehicle_id
        }

    @staticmethod
    def _parse_vehicle_id(path):
        "First part of the file path gives the vehicle ID"
        vehicle_id = path.stem.split('_')[0]
        return int(vehicle_id)

# FUNCTIONS TO TRANSFORM IMAGE DATA FOR TRAINING AND EVAL
def train_transforms(image_resolution):
    return transforms .Compose([
        transforms.Resize((image_resolution, image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],  # ImageNet means
            std = [0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])

def eval_transforms(image_resolution):
    return transforms.Compose([
        transforms.Resize((image_resolution, image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])

# FUNCTION TO CREATE CALL DATASET CLASS
def create_datasets(data_dir, image_resolution):
    """
    This function creates training and eval datasets from the raw data directory.
    """

    root_dir = Path(data_dir)
    train_dir = root_dir / 'image_train'
    query_dir = root_dir / 'image_query'
    test_dir = root_dir / 'image_test'

    # Assemble datasets
    train_dataset = VeRiDataset_Train(train_dir, transform=train_transforms(image_resolution))
    query_dataset = VeRiDataset_Eval(query_dir, transform=eval_transforms(image_resolution))
    gallery_dataset = VeRiDataset_Eval(test_dir, transform=eval_transforms(image_resolution))

    return train_dataset, query_dataset, gallery_dataset

# FUNCTION TO CREATE DATALOADERS
def create_dataloaders(
        data_dir, image_resolution, batch_size=64
    ):
    """
    This function creates dataloaders for the training and eval datasets.
    """

    # Create datasets
    train_dataset, query_dataset, gallery_dataset = create_datasets(
        data_dir, 
        image_resolution
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        persistent_workers=True
    )
    query_loader = DataLoader(
        query_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )
    gallery_loader = DataLoader(
        gallery_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )

    return train_loader, query_loader, gallery_loader
