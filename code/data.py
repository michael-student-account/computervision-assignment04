# A HELPER FILE FOR CLASSES AND FUNCTIONS THAT IMPORT AND TRANSFORM THE DATASET
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

# CUSTOM DATASET CLASS
class VeRiDataset(Dataset):
    """
    A custom dataset class for the VeRi dataset that translates the raw 
    datset folder into a useable format for PyTorch dataloaders.
    Adapted from a shell structure suggested by ChatGPT.
    """
    def __init__(self, image_dir, transform=None):
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
def train_transforms():
    return transforms .Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],  # ImageNet means
            std = [0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])

def eval_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])

# FUNCTION TO CREATE CALL DATASET CLASS
def create_datasets(data_dir):
    """
    This function creates training and eval datasets from the raw data directory
    """

    root_dir = Path(data_dir)
    train_dir = root_dir / 'image_train'
    query_dir = root_dir / 'image_query'
    test_dir = root_dir / 'image_test'

    # Assemble datasets
    train_dataset = VeRiDataset(train_dir, transform=train_transforms())
    query_dataset = VeRiDataset(query_dir, transform=eval_transforms())
    gallery_dataset = VeRiDataset(test_dir, transform=eval_transforms())

    return train_dataset, query_dataset, gallery_dataset

# FUNCTION TO CREATE DATALOADERS
def create_dataloaders(
        train_dataset, query_dataset, gallery_dataset, batch_size=64
    ):
    """
    This function creates dataloaders for the training and eval datasets.
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    query_loader = DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, query_loader, gallery_loader



