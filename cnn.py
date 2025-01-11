#CNN based
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from feature_extraction import extract_features
from data_loader import load_data, CatDogDataset
from evaluation import find_similar_images_knn, evaluate_similarity
from models import get_resnet50
from common_training import evaluate_model



# --- DATA LOADING AND PREPROCESSING ---
data_dir = "/content/PetImages"
train_paths, test_paths, train_labels, test_labels = load_data(data_dir)
train_dataset = CatDogDataset(train_paths, train_labels, data_transforms['train'])
test_dataset = CatDogDataset(test_paths, test_labels, data_transforms['test'])
dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    }
dataset_sizes = {'train':len(train_dataset),
'test' : len(test_dataset)}



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



    

# --- MODEL DEFINITION AND TRAINING ---
model = get_resnet50()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 1
for epoch in range(num_epochs):
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        all_labels = []
        all_preds = []

        for inputs, labels in tqdm(dataloaders[phase], desc=f"Epoch {epoch+1}/{num_epochs} - {phase}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1:.4f}')



# --- FEATURE EXTRACTION FOR ALL TEST IMAGES (Done ONCE) ---
all_test_features = []
for test_image_path in tqdm(test_paths, desc="Extracting features for all test images"):
    features = extract_features(test_image_path, model, data_transforms['test'], device)
    if features is not None:
        all_test_features.append(features)

if not all_test_features:
    raise ValueError("No valid features extracted from test set.")

all_test_features = np.concatenate(all_test_features, axis=0)

# --- EVALUATION ON TEST SET (using KNN) ---

evaluate_model(model, test_paths, transform, device, extract_features)

# Example Usage (after training and evaluation - using KNN)
example_image_path = test_paths[0]
example_features = all_test_features[0].reshape(1, -1)
similar_images = find_similar_images_knn(example_features, all_test_features, test_paths)

if similar_images:
    print(f"\nSimilar images to {example_image_path}:")
    for image_path, similarity in similar_images:
        print(f"- {image_path}: Similarity = {similarity:.4f}")
else:
    print("Could not find similar images.")