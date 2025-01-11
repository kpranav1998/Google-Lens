#Auto enconder
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
from models import Autoencoder
from common_training import evaluate_model


# --- DATA LOADING AND PREPROCESSING ---

data_dir = "/content/PetImages"
train_paths, test_paths, train_labels, test_labels = load_data(data_dir)



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([ # Simplified test transform
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = CatDogDataset(train_paths, train_labels, data_transforms['train'])
test_dataset = CatDogDataset(test_paths, test_labels, data_transforms['test'])
dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    }
dataset_sizes = {'train':len(train_dataset),
'test' : len(test_dataset)}

# --- MODEL DEFINITION AND TRAINING ---

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# --- Autoencoder Training ---
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, _ in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")



# --- FEATURE EXTRACTION FOR ALL TEST IMAGES (Done ONCE) ---
all_test_features = []
for test_image_path in tqdm(test_paths, desc="Extracting features for all test images"):
    features = extract_features(test_image_path, model, data_transforms['test'], device)
    if features is not None:
        all_test_features.append(features)

if not all_test_features:
    raise ValueError("No valid features extracted from test set.")

all_test_features = np.concatenate(all_test_features, axis=0)

# --- Similarity Search (using KNN) ---
num_similar_images = 5 # Number of similar images to retrieve
num_test_images = 5000 # Number of test images to use for example and testing
test_subset_paths = test_paths[:num_test_images]
test_subset_features = all_test_features[:num_test_images]

# Example Usage
example_image_path = test_subset_paths[0]
example_features = test_subset_features[0].reshape(1, -1)

similar_images = find_similar_images_knn(example_features, all_test_features, test_paths, top_k=num_similar_images)

if similar_images:
    print(f"\nSimilar images to {example_image_path}:")
    for image_path, similarity in similar_images:
        print(f"- {image_path}: Similarity = {similarity:.4f}")
else:
    print("Could not find similar images.")

# Test on a subset of the test data
print("\nTesting on a subset of test data")
for i, query_image_path in tqdm(enumerate(test_subset_paths), total=len(test_subset_paths), desc="Finding similar images"):
    query_features = test_subset_features[i].reshape(1, -1)
    similar_images = find_similar_images_knn(query_features, all_test_features, test_paths, top_k=num_similar_images)
    if similar_images:
        print(f"\nSimilar images to {query_image_path}:")
        for image_path, similarity in similar_images:
            print(f"- {image_path}: Similarity = {similarity:.4f}")
    else:
        print("Could not find similar images.")