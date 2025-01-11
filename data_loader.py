# data_loader.py
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class CatDogDataset(Dataset):
    def __init__(self, image_paths, labels,transforms):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(data_dir, test_size=0.2):
    image_paths = []
    labels = []

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            label = 1 if class_name.lower() == "dog" else 0
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_path = os.path.join(class_dir, filename)
                    try:
                        Image.open(image_path).convert('RGB')
                        image_paths.append(image_path)
                        labels.append(label)
                    except:
                        print(f"Invalid image: {image_path}. Skipping.")

    if not image_paths:
        raise FileNotFoundError(f"No valid images found in dataset directory: {data_dir}")

    return train_test_split(image_paths, labels, test_size=test_size, random_state=42, stratify=labels)