import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, tv_tensors
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io
import xml.etree.ElementTree as ET
from torchvision.tv_tensors import BoundingBoxes, Mask
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pet dataset class
class PetDataset(Dataset):
# base_dir = ../The_Oxford-IIIT_Pet_Dataset
    def __init__(self, data_dir, image_ids, labels, transform=None):
        self.root = data_dir
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        label = self.labels[idx]
        image_path = os.path.join(self.root, 'images', f"{image_name}.jpg")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
        # # Load XML annotations
        # xml_path = os.path.join(self.root, 'annotations', 'xmls', f"{image_name}.xml")
        # tree = ET.parse(xml_path)
        # root = tree.getroot()

        # # Get image dimensions
        # size = root.find('size')
        # width = int(size.find('width').text)
        # height = int(size.find('height').text)
        
        # boxes = []
        # labels = []
        # iscrowd = []
        # for obj in root.findall('object'):
        #     bbox = obj.find('bndbox')
        #     xmin = int(bbox.find('xmin').text)
        #     ymin = int(bbox.find('ymin').text)
        #     xmax = int(bbox.find('xmax').text)
        #     ymax = int(bbox.find('ymax').text)
        #     boxes.append([xmin, ymin, xmax, ymax])
        #     labels.append(label)
        #     iscrowd.append(0)  # All instances are not crowd
        
        # boxes = torch.tensor(boxes, dtype=torch.float32)
        # labels = torch.tensor(labels, dtype=torch.int64)
        # iscrowd = torch.tensor(iscrowd, dtype=torch.uint8)
        
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # # Load masks
        # mask_path = os.path.join(self.root, 'annotations', 'trimaps', f"{image_name}.png")
        # mask = Image.open(mask_path)
        # mask_np = np.array(mask)
        # # Convert trimap to binary mask (1=foreground, 2=background, 3=unclear)
        # mask_np = (mask_np == 1).astype(np.uint8)
        # masks = torch.tensor(mask_np, dtype=torch.uint8).unsqueeze(0)  # Add instance dimension
        # masks = tv_tensors.Mask(masks)

        # target = {
        #     "boxes": BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width)),
        #     "labels": labels,
        #     "image_id": torch.tensor([idx]),  # Unique ID for evaluation
        #     "area": area,
        #     "iscrowd": iscrowd,
        #     # "masks": masks
        # }
            
        # if self.transform:
        #     image, target = self.transform(image, target)   

        # return image, target

    
# Checked
# Parse the dataset list file
def parse_list_file(base_dir):
    list_path = os.path.join(base_dir, "annotations", "list.txt")
    
    image_ids = []
    labels_multiclass = []  #ID: 1:37 Class ids
    label_species = []  #SPECIES: 1:Cat 2:Dog
    label_breeds = [] #BREED ID: 1-25:Cat 1:12:Dog
    
    with open(list_path, 'r') as f:
        # Skip the header lines
        list_lines = [line.strip() for line in f if not line.startswith('#')]
            
    for line in list_lines:
        if not line: continue

        parts = line.strip().split()
        if len(parts) >= 4:  # Ensure we have all needed fields
            image_name = parts[0]
            class_id = int(parts[1]) - 1
            species = int(parts[2]) - 1  # 1: Cat, 2: Dog
            breeds = int(parts[3]) - 1
            
            image_ids.append(image_name)
            labels_multiclass.append(class_id)
            label_species.append(species)
            label_breeds.append(breeds)
    
    return image_ids, labels_multiclass, label_species, label_breeds
    
    # Create data loaders

def create_data_loaders(data_dir, image_ids, labels, batch_size=32, train_transform = None):

    # Train/validation/test split (70%/15%/15%)
    train_image_ids, temp_image_ids, train_labels, temp_labels = train_test_split(
        image_ids, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_image_ids, test_image_ids, val_labels, test_labels = train_test_split(
        temp_image_ids, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Define transformations
    if train_transform == None:
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
    train_dataset = PetDataset(data_dir, train_image_ids, train_labels, train_transform)
    val_dataset = PetDataset(data_dir, val_image_ids, val_labels, test_transform)
    test_dataset = PetDataset(data_dir, test_image_ids, test_labels, test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# Initialize and modify ResNet model
def initialize_model(num_classes, unfreeze_layers=0, feature_extract=True):
    # Load pre-trained ResNet18
    # model = models.resnet18(weights='IMAGENET1K_V1')
    # model = models.resnet34(weights='IMAGENET1K_V1') 
    model = models.resnet50(weights='IMAGENET1K_V2') 
    
    # Set to feature extraction mode (only train the final layer)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # Unfreeze layers
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    for layer in layers[:unfreeze_layers]:
        for param in layer.parameters():
            param.requires_grad = True
    
    # Replace the final FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, scheduler=None):
    
    model = model.to(device)

    # Track history
    loss_and_accs = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        # Load data
        model.train()

        running_loss = 0.0
        running_corrects = 0
        total = 0

        # Iterate over data
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss * 1.0 / total
        epoch_train_acc = running_corrects * 1.0 / total

        epoch_val_loss, epoch_val_acc = evaluate_model(model, val_loader, criterion)
        if scheduler:
            scheduler.step(epoch_val_loss)

        loss_and_accs['train_loss'].append(epoch_train_loss)
        loss_and_accs['train_acc'].append(epoch_train_acc)
        loss_and_accs['val_loss'].append(epoch_val_loss)
        loss_and_accs['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")

    torch.cuda.empty_cache()
    return model, loss_and_accs

def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()
    
    loss = running_loss * 1.0 / total
    acc = running_corrects * 1.0 / total
    return loss, acc

# Visualize training history
def plot_training_history(loss_and_accs):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_and_accs['train_loss'], label='Training Loss')
    plt.plot(loss_and_accs['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(loss_and_accs['train_acc'], label='Training Accuracy')
    plt.plot(loss_and_accs['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def classification_for_2():
    data_dir = "../The_Oxford-IIIT_Pet_Dataset"  # Update path if needed
    image_ids, _, label_species, _ = parse_list_file(data_dir)
    
    # Create data loaders for species classification (binary)
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, image_ids, label_species, batch_size=64
    )
    
    # Initialize model for binary classification
    model = initialize_model(num_classes=2, unfreeze_layers=0, feature_extract=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Train the model
    model, loss_and_accs = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    
    # Plot training 
    plot_training_history(loss_and_accs)

    # Final evaluation on test set
    _, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'pet_binary_model.pth')

# Strategy 1
def classification_for_37(num_epochs=25, train_transform=None, learning_rates=[0.001, 0.0005, 0.0001, 0.00005, 0.00001], unfreeze_layers=0):
    for l in range(unfreeze_layers):
        data_dir = "../The_Oxford-IIIT_Pet_Dataset"  # Update path if needed
        image_ids, labels_multiclass, _, _ = parse_list_file(data_dir)
        
        # Create data loaders for species classification (binary)
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir, image_ids, labels_multiclass, batch_size=32, train_transform=train_transform
        )
        
        # Initialize model for binary classification
        model = initialize_model(num_classes=37, unfreeze_layers=l, feature_extract=True)
        model = model.to(device)
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        optimizer_prams = [
            {'params': model.fc.parameters(), 'lr': learning_rates[0], 'weight_decay': 1e-4}
        ]
        for l in range(unfreeze_layers):
            optimizer_prams.append({'params': layers[l].parameters(), 'lr': learning_rates[l+1], 'weight_decay': 1e-4})

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(optimizer_prams[:l+1])
        
        # Train the model
        model, loss_and_accs = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
        
        # Plot training 
        plot_training_history(loss_and_accs)

        # Final evaluation on test set
        _, test_acc = evaluate_model(model, test_loader, criterion)
        print(f"Test Accuracy with unfreeze_layers={l}: {test_acc:.4f}")

        # Save the model
        torch.save(model.state_dict(), f'pet_breed_model_l{l}.pth')


# Strategy 2
def classification_for_37_strategy_2(
        learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001], 
        weight_decay=0, unfreeze_layers=4, train_transform=None, batch_size=32,
        num_epochs = [10, 10, 10, 10, 10]):
    data_dir = "../The_Oxford-IIIT_Pet_Dataset"  # Update path if needed
    image_ids, labels_multiclass, _, _ = parse_list_file(data_dir)
    
    # Create data loaders for species classification (binary)
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, image_ids, labels_multiclass, batch_size=batch_size, train_transform=train_transform
    )
    
    # Initialize model for binary classification
    model = initialize_model(num_classes=37, feature_extract=True)
    model = model.to(device)

    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    all_loss_and_accs = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Train the model
    # optimizer = optim.Adam(model.fc.parameters(), lr=learning_rates[0], weight_decay=weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rates[0], weight_decay=weight_decay)
    model, loss_and_accs = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs[0])
    all_loss_and_accs['train_loss'].extend(loss_and_accs['train_loss'])
    all_loss_and_accs['val_loss'].extend(loss_and_accs['val_loss'])
    all_loss_and_accs['train_acc'].extend(loss_and_accs['train_acc'])
    all_loss_and_accs['val_acc'].extend(loss_and_accs['val_acc'])

    _, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"Test Accuracy with unfreeze_layers={0}: {test_acc:.4f}\n")

    for l in range(unfreeze_layers):
        for param in layers[l].parameters():
            param.requires_grad = True  
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rates[l+1], weight_decay=weight_decay)
        model, loss_and_accs = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs[l+1])
        all_loss_and_accs['train_loss'].extend(loss_and_accs['train_loss'])
        all_loss_and_accs['val_loss'].extend(loss_and_accs['val_loss'])
        all_loss_and_accs['train_acc'].extend(loss_and_accs['train_acc'])
        all_loss_and_accs['val_acc'].extend(loss_and_accs['val_acc'])

        _, test_acc = evaluate_model(model, test_loader, criterion)
        print(f"Test Accuracy with unfreeze_layers={l+1}: {test_acc:.4f}\n")
        torch.save(model.state_dict(), f'pet_breed_model_l{l}.pth')
    
    
    # # Plot training 
    plot_training_history(all_loss_and_accs)

def classification_for_37_strategy_2_learning_rates_scheduler(
        learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001], 
        weight_decay=0, unfreeze_layers=4, train_transform=None, batch_size=32,
        num_epochs = [10, 10, 10, 10, 10]):
    data_dir = "../The_Oxford-IIIT_Pet_Dataset"  # Update path if needed
    image_ids, labels_multiclass, _, _ = parse_list_file(data_dir)
    
    # Create data loaders for species classification (binary)
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, image_ids, labels_multiclass, batch_size=batch_size, train_transform=train_transform
    )
    
    # Initialize model for binary classification
    model = initialize_model(num_classes=37, feature_extract=True)
    model = model.to(device)

    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    all_loss_and_accs = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Train the model
    # optimizer = optim.Adam(model.fc.parameters(), lr=learning_rates[0], weight_decay=weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rates[0], weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',    # Monitor validation loss
        factor=0.1,    # Reduce LR by 10x
        patience=3,    # Wait 3 epochs w/o improvement
    )
    model, loss_and_accs = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs[0], scheduler=scheduler)
    all_loss_and_accs['train_loss'].extend(loss_and_accs['train_loss'])
    all_loss_and_accs['val_loss'].extend(loss_and_accs['val_loss'])
    all_loss_and_accs['train_acc'].extend(loss_and_accs['train_acc'])
    all_loss_and_accs['val_acc'].extend(loss_and_accs['val_acc'])

    _, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"Test Accuracy with unfreeze_layers={0}: {test_acc:.4f}\n")

    for l in range(unfreeze_layers):
        for param in layers[l].parameters():
            param.requires_grad = True  
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rates[l+1], weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',    # Monitor validation loss
            factor=0.1,    # Reduce LR by 10x
            patience=3,    # Wait 3 epochs w/o improvement
        )
        model, loss_and_accs = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs[l+1], scheduler=scheduler)
        all_loss_and_accs['train_loss'].extend(loss_and_accs['train_loss'])
        all_loss_and_accs['val_loss'].extend(loss_and_accs['val_loss'])
        all_loss_and_accs['train_acc'].extend(loss_and_accs['train_acc'])
        all_loss_and_accs['val_acc'].extend(loss_and_accs['val_acc'])

        _, test_acc = evaluate_model(model, test_loader, criterion)
        print(f"Test Accuracy with unfreeze_layers={l+1}: {test_acc:.4f}\n")
        torch.save(model.state_dict(), f'classification_for_37_strategy_2_learning_rates_scheduler_pet_breed_model_l{l}.pth')
    
    
    # # Plot training 
    plot_training_history(all_loss_and_accs)


