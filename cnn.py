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
data_dir = "/kaggle/input/autoencoder/archive/PetImages/"
train_paths, test_paths, train_labels, test_labels = load_data(data_dir)
ground_truth_labels = {path: label for path, label in zip(test_paths, test_labels)}

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

train_dataset = CatDogDataset(train_paths, train_labels, data_transforms['train'])
test_dataset = CatDogDataset(test_paths, test_labels, data_transforms['test'])

dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    }
dataset_sizes = {'train':len(train_dataset),
'test' : len(test_dataset)}

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

def evaluate_similarity(query_image_path, retrieved_images, ground_truth_labels, k):
    """Calculates Precision@k and Recall@k for a single query."""
    relevant_count = 0
    for image_path, _ in retrieved_images[:k]:
        if ground_truth_labels.get(image_path) is not None and ground_truth_labels.get(query_image_path) is not None and ground_truth_labels[image_path] == ground_truth_labels[query_image_path]:
            relevant_count += 1

    precision_at_k = relevant_count / k if k > 0 else 0.0
    total_relevant = sum(1 for path, label in ground_truth_labels.items() if ground_truth_labels.get(path) is not None and label == ground_truth_labels[query_image_path])
    recall_at_k = relevant_count / total_relevant if total_relevant > 0 else 0.0
    return precision_at_k, recall_at_k

def calculate_map(test_paths, all_test_features, ground_truth_labels, k, data_transforms, model, device):
    """Calculates Mean Average Precision (MAP) for the entire test set."""
    ap_scores = []
    for i, query_image_path in enumerate(tqdm(test_paths, desc="Calculating MAP")):
        query_features = all_test_features[i].reshape(1, -1)
        retrieved_images = find_similar_images_knn(query_features, all_test_features, test_paths)
        precision_at_k, recall_at_k = evaluate_similarity(query_image_path, retrieved_images, ground_truth_labels, k)
        ap_scores.append(precision_at_k)

    map_score = np.mean(ap_scores) if ap_scores else 0.0
    return map_score


#evaluate_model(model, test_paths, transform, device, extract_features)


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


k = 5  # Example value for k
precision_at_k, recall_at_k = evaluate_similarity(example_image_path, similar_images, ground_truth_labels, k)
map_score = calculate_map(test_paths, all_test_features, ground_truth_labels, k, data_transforms, model, device)


print(f"Precision@{k}: {precision_at_k:.4f}")
print(f"Recall@{k}: {recall_at_k:.4f}")
print(f"MAP@{k}: {map_score:.4f}")