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
from models import ContrastiveLoss,Autoencoder
from common_training import evaluate_model
import torch.nn.functional as F

import torch
import torch.nn as nn
class SiameseNetwork(nn.Module):
    def __init__(self, input_size=(224, 224)):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.ReLU(inplace=True)
        )

        # Calculate CNN output size dynamically
        test_input = torch.randn(1, 3, input_size[0], input_size[1])
        with torch.no_grad():
            cnn_output = self.cnn(test_input)
            cnn_output_size = cnn_output.view(cnn_output.size(0), -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_output_size, 4096),
            nn.Sigmoid()
        )

        def forward_once(self, x):
            x = self.cnn(x)
            #x = self.fc(x)
            return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


    def forward_once(self, x):
        x = self.cnn(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """Compute contrastive loss."""
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss
data_dir = "/kaggle/input/autoencoder/archive/PetImages/"
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
            #print(f"Input image 1 shape: {img1.shape}")
            #print(f"Input image 2 shape: {img2.shape}")
            target = torch.tensor([target], dtype=torch.float32).to(device)  # Convert int to float tensor
            
            output1, output2 = net(img1, img2)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, target)
            loss_contrastive.backward()
            optimizer.step()
            
            epoch_loss += loss_contrastive.item()
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(training_pairs):.4f}")


# --- MODEL DEFINITION AND TRAINING ---

num_epochs = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

train_siamese(model, criterion, optimizer, training_pairs, data_transforms['train'], device, epochs=num_epochs)

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity  # Import here

def extract_features_siamese(image_path, model, transform, device):
  """Extracts features from a single image using a Siamese Network."""
  try:
      image = Image.open(image_path).convert('RGB')
  except FileNotFoundError:
      print(f"Error: Image not found at {image_path}")
      return None
  except Exception as e:
      print(f"Error opening image: {e}")
      return None

  image_tensor = transform(image).unsqueeze(0).to(device)
  model.eval()
  with torch.no_grad():
      # Forward pass through the Siamese network (requires two inputs)
      features = model(image_tensor, image_tensor)  # Pass the image twice
      features = features[0]
      features = torch.flatten(features, 1)  # Flatten the output

  return features
    
def calculate_similarity(output1, output2, metric="euclidean"):
    """Calculates similarity between two feature vectors."""
    if metric == "euclidean":
        return euclidean_distances(output1, output2)
    elif metric == "cosine":
        return cosine_similarity(output1, output2)
    else:
        raise ValueError("Invalid similarity metric. Choose 'euclidean' or 'cosine'.")

def find_similar_images(query_features, all_features, all_paths, top_k=5, metric="euclidean"):
    """Finds the top k most similar images using pre-computed features."""

    if metric == "euclidean":
        similarities = calculate_similarity(query_features, all_features)
        # Get indices of top k smallest distances (most similar)
        indices = np.argsort(similarities[0])[:top_k]
    elif metric == "cosine":
        similarities = calculate_similarity(query_features, all_features)
        # Get indices of top k largest similarities (most similar)
        indices = np.argsort(similarities[0])[::-1][:top_k]
    else:
        raise ValueError("Invalid similarity metric. Choose 'euclidean' or 'cosine'.")
    
    similar_images = [(all_paths[i], similarities[0][i]) for i in indices]
    return similar_images

# --- FEATURE EXTRACTION FOR ALL TEST IMAGES (Done ONCE) ---
all_test_features = []
for test_image_path in tqdm(test_paths, desc="Extracting features for all test images"):
    features = extract_features_siamese(test_image_path, model, data_transforms['test'], device)
    if features is not None:
        all_test_features.append(features)

if not all_test_features:
    raise ValueError("No valid features extracted from test set.")

all_test_features = [f.cpu().numpy() for f in all_test_features]  # Move each feature to CPU

all_test_features = np.concatenate(all_test_features, axis=0)

ground_truth_labels = {path: label for path, label in zip(test_paths, test_labels)}

# --- Similarity Search ---
num_similar_images = 5
num_test_images = 100
test_subset_paths = test_paths[:num_test_images]
test_subset_features = all_test_features[:num_test_images]

# Example Usage (using the new function)
example_image_path = test_subset_paths[0]
example_features = all_test_features[0].reshape(1, -1)

similar_images = find_similar_images(example_features, all_test_features, test_paths, top_k=num_similar_images, metric = "euclidean")

if similar_images:
    print(f"\nSimilar images to {example_image_path}:")
    for image_path, similarity in similar_images:
        print(f"- {image_path}: Similarity = {similarity:.4f}")
else:
    print("Could not find similar images.")

# Test on a subset of the test data (using the new function)
'''print("\nTesting on a subset of test data")
for i, query_image_path in tqdm(enumerate(test_subset_paths), total=len(test_subset_paths), desc="Finding similar images"):
    query_features = test_subset_features[i].reshape(1, -1)
    similar_images = find_similar_images(query_features, all_test_features, test_paths, top_k=num_similar_images, metric = "euclidean")
    if similar_images:
        #print(f"\nSimilar images to {query_image_path}:")
        for image_path, similarity in similar_images:
            print(f"- {image_path}: Similarity = {similarity:.4f}")
    else:
        print("Could not find similar images.")
'''


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

def calculate_map(test_paths, all_test_features, ground_truth_labels, k, find_similar_images):
    """Calculates Mean Average Precision (MAP)."""
    ap_scores = []
    for i, query_image_path in enumerate(tqdm(test_paths, desc="Calculating MAP")):
        query_features = all_test_features[i].reshape(1, -1)
        retrieved_images = find_similar_images(query_features, all_test_features, test_paths, top_k=k)
        precision_at_k, _ = evaluate_similarity(query_image_path, retrieved_images, ground_truth_labels, k)
        ap_scores.append(precision_at_k)  # Using precision@k as a proxy for AP here (simplification)

    map_score = np.mean(ap_scores) if ap_scores else 0.0
    return map_score

# ... (Your feature extraction and similarity search code)

# Example Usage (after feature extraction and similarity search)
k = 5  # Example value for k

precision_at_k_values = []
recall_at_k_values = []

for i, query_image_path in enumerate(tqdm(test_subset_paths, desc="Evaluating Similarity")):
    query_features = test_subset_features[i].reshape(1, -1)
    similar_images = find_similar_images(query_features, all_test_features, test_paths, top_k=k, metric="euclidean")
    precision_at_k, recall_at_k = evaluate_similarity(query_image_path, similar_images, ground_truth_labels, k)
    precision_at_k_values.append(precision_at_k)
    recall_at_k_values.append(recall_at_k)

mean_precision_at_k = np.mean(precision_at_k_values)
mean_recall_at_k = np.mean(recall_at_k_values)
map_score = calculate_map(test_subset_paths, test_subset_features, ground_truth_labels, k, find_similar_images)

print(f"Mean Precision@{k}: {mean_precision_at_k:.4f}")
print(f"Mean Recall@{k}: {mean_recall_at_k:.4f}")
print(f"MAP@{k}: {map_score:.4f}")