#Siamese Network
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
from models import SiameseNetwork,ContrastiveLoss
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

def create_pairs(paths, labels):
    pairs = []
    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}  # Map labels to indices
    digit_indices = [np.where(labels == label)[0] for label in unique_labels]  # Find indices for each label
    
    for d in range(len(paths)):
        img1_idx = d
        x1 = paths[img1_idx]
        label1 = labels[img1_idx]
        mapped_label1 = label_to_index[label1]  # Get mapped label index
        
        same_class_indices = digit_indices[mapped_label1]
        siamese_label = np.random.randint(0, 2)
        
        if siamese_label == 1:
            img2_idx = same_class_indices[np.random.randint(0, len(same_class_indices))]
            x2 = paths[img2_idx]
        else:
            diff_class_indices = np.where(labels != label1)[0]
            img2_idx = diff_class_indices[np.random.randint(0, len(diff_class_indices))]
            x2 = paths[img2_idx]
        
        pairs.append((x1, x2, siamese_label))  # Use tuple, not array
    
    return pairs  

training_pairs = create_pairs(train_paths, train_labels)


def load_image(path, transform):
    image = Image.open(path).convert('RGB')
    image = transform(image)
    return image.to(device)

def train_siamese(net, criterion, optimizer, training_pairs, transform, device, epochs=10):
    for epoch in range(epochs):
        epoch_loss = 0.0
        for pair in tqdm(training_pairs, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            img1_path, img2_path, target = pair
            img1 = load_image(img1_path, transform)
            img2 = load_image(img2_path, transform)
            target = torch.tensor([target], dtype=torch.float32).to(device)  # Convert int to float tensor
            
            output1, output2 = net(img1, img2)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, target)
            loss_contrastive.backward()
            optimizer.step()
            
            epoch_loss += loss_contrastive.item()
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(training_pairs):.4f}")


# --- MODEL DEFINITION AND TRAINING ---

num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

train_siamese(model, criterion, optimizer, training_pairs, data_transforms['train'], device, epochs=num_epochs)

# --- FEATURE EXTRACTION FOR ALL TEST IMAGES (Done ONCE) ---
all_test_features = []
for test_image_path in tqdm(test_paths, desc="Extracting features for all test images"):
    features = extract_features(test_image_path, net, data_transforms['test'], device)
    if features is not None:
        all_test_features.append(features)

if not all_test_features:
    raise ValueError("No valid features extracted from test set.")

all_test_features = np.concatenate(all_test_features, axis=0)

# --- Similarity Search (using KNN) ---
num_similar_images = 5
num_test_images = 100
test_subset_paths = test_paths[:num_test_images]
test_subset_features = all_test_features[:num_test_images]

# Example Usage
example_image_path = test_subset_paths[0]
example_features = all_test_features[0].reshape(1, -1)

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