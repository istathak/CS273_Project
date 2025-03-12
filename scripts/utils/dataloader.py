import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def load_data(train_dir, batch_size=32, val_split=0.2):
    #transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((224, 224)),  # Resize images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])
    
    
    full_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    labels = [label for _, label in full_dataset.samples]  # Extract labels from dataset
    
    # Train-validation split (stratification)
    train_idx, val_idx = train_test_split(
        list(range(len(labels))), test_size=val_split, stratify=labels, random_state=42
    )
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    #DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Data Class
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Total images: {len(full_dataset)} | Training: {len(train_dataset)} | Validation: {len(val_dataset)}")

    return train_loader, val_loader



