import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io
import time

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download and extract the Oxford-IIIT Pet Dataset
def download_pet_dataset():
    base_dir = "oxford-iiit-pet"
    
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
        # Download images
        print("Downloading images...")
        image_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        r = requests.get(image_url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(base_dir)
        
        # Download annotations
        print("Downloading annotations...")
        anno_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
        r = requests.get(anno_url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(base_dir)
        
    return base_dir

# Parse the dataset list file
def parse_list_file(base_dir):
    list_path = os.path.join(base_dir, "annotations", "list.txt")
    
    images = []
    labels_binary = []  # Dog vs Cat (0: Dog, 1: Cat)
    labels_multiclass = []  # 37 breeds
    
    with open(list_path, 'r') as f:
        # Skip the header lines
        for _ in range(6):
            next(f)
            
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:  # Ensure we have all needed fields
                image_name = parts[0]
                class_id = int(parts[1]) - 1  # Convert to 0-indexed
                species = int(parts[2])  # 1: Cat, 2: Dog
                
                image_path = os.path.join(base_dir, "images", f"{image_name}.jpg")
                if os.path.exists(image_path):
                    images.append(image_path)
                    labels_binary.append(1 if species == 1 else 0)  # 1 for Cat, 0 for Dog
                    labels_multiclass.append(class_id)
    
    return images, labels_binary, labels_multiclass

# Pet dataset class
class PetDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Create data loaders
def create_data_loaders(images, labels, batch_size=32, task="binary"):
    # Train/validation/test split (70%/15%/15%)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = PetDataset(train_images, train_labels, train_transform)
    val_dataset = PetDataset(val_images, val_labels, test_transform)
    test_dataset = PetDataset(test_images, test_labels, test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

# Initialize and modify ResNet model
def initialize_model(num_classes, feature_extract=True):
    # Load pre-trained ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Set to feature extraction mode (only train the final layer)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    model = model.to(device)
    
    # Track best model and metrics
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Track history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch}/{num_epochs}')
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if it's the best
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
        
        print()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

# Evaluate model on test set
def evaluate_model(model, test_loader):
    model.eval()
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
    
    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    return test_acc

# Visualize training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Main execution
def main():
    # Download and extract the dataset
    base_dir = download_pet_dataset()
    
    # Parse the list file
    images, labels_binary, labels_multiclass = parse_list_file(base_dir)
    print(f"Total images found: {len(images)}")
    print(f"Binary labels distribution: Cats={labels_binary.count(1)}, Dogs={labels_binary.count(0)}")
    
    # Part 1: Binary Classification (Dog vs Cat)
    print("\n=== Part 1: Binary Classification (Dog vs Cat) ===")
    
    # Create data loaders for binary classification
    train_loader_binary, val_loader_binary, test_loader_binary = create_data_loaders(
        images, labels_binary, batch_size=32, task="binary"
    )
    
    # Initialize the model for binary classification
    model_binary = initialize_model(num_classes=2, feature_extract=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_binary = optim.Adam(model_binary.fc.parameters(), lr=0.001)
    
    # Train the model
    print("Training binary classification model...")
    model_binary, history_binary = train_model(
        model_binary, train_loader_binary, val_loader_binary, criterion, optimizer_binary, num_epochs=10
    )
    
    # Evaluate the model
    print("Evaluating binary classification model...")
    binary_test_acc = evaluate_model(model_binary, test_loader_binary)
    
    # Save the model
    torch.save(model_binary.state_dict(), 'pet_binary_model.pth')
    
    # Plot training history
    plot_training_history(history_binary)
    
    # Part 2: Multi-class Classification (37 Breeds)
    print("\n=== Part 2: Multi-class Classification (37 Breeds) ===")
    
    # Create data loaders for multi-class classification
    train_loader_multi, val_loader_multi, test_loader_multi = create_data_loaders(
        images, labels_multiclass, batch_size=32, task="multiclass"
    )
    
    # Initialize the model for multi-class classification
    model_multi = initialize_model(num_classes=37, feature_extract=True)
    
    # Define loss function and optimizer
    optimizer_multi = optim.Adam(model_multi.fc.parameters(), lr=0.001)
    
    # Train the model
    print("Training multi-class classification model (final layer only)...")
    model_multi, history_multi_fc = train_model(
        model_multi, train_loader_multi, val_loader_multi, criterion, optimizer_multi, num_epochs=15
    )
    
    # Evaluate the model
    print("Evaluating multi-class classification model (final layer only)...")
    multi_test_acc_fc = evaluate_model(model_multi, test_loader_multi)
    
    # Strategy 1: Fine-tune the last few layers
    print("\nStrategy 1: Fine-tuning the last few layers...")
    
    # Unfreeze the last layer group (layer4) and final fc layer
    for param in model_multi.layer4.parameters():
        param.requires_grad = True
    
    # New optimizer for all trainable parameters
    optimizer_multi_ft = optim.Adam([
        {'params': model_multi.fc.parameters(), 'lr': 0.001},
        {'params': model_multi.layer4.parameters(), 'lr': 0.0001}
    ])
    
    # Train the model with more layers unfrozen
    print("Training multi-class classification model (layer4 + FC)...")
    model_multi, history_multi_ft = train_model(
        model_multi, train_loader_multi, val_loader_multi, criterion, optimizer_multi_ft, num_epochs=15
    )
    
    # Evaluate the model
    print("Evaluating multi-class classification model (layer4 + FC)...")
    multi_test_acc_ft = evaluate_model(model_multi, test_loader_multi)
    
    # Save the model
    torch.save(model_multi.state_dict(), 'pet_multiclass_model.pth')
    
    # Plot training history
    plot_training_history(history_multi_ft)
    
    # Print summary
    print("\n=== Results Summary ===")
    print(f"Binary Classification Test Accuracy: {binary_test_acc:.4f}")
    print(f"Multi-class Classification (FC only) Test Accuracy: {multi_test_acc_fc:.4f}")
    print(f"Multi-class Classification (Layer4 + FC) Test Accuracy: {multi_test_acc_ft:.4f}")

if __name__ == "__main__":
    main()
