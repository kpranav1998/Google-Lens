# common_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from evaluation import find_similar_images_knn, evaluate_similarity



def evaluate_model(model, test_paths, transform, device, extract_features_func):
    print("\n--- Extracting features for all test images ---")
    all_test_features = []
    for test_image_path in tqdm(test_paths, desc="Extracting features"):
        features = extract_features_func(test_image_path, model, transform, device)
        if features is not None:
            all_test_features.append(features)

    if not all_test_features:
        raise ValueError("No valid features extracted from test set.")

    all_test_features = np.concatenate(all_test_features, axis=0)


    print("\n--- Evaluating on test set ---")
    num_test_images = min(100, len(test_paths))  # Limit evaluation to 100 images
    test_subset_paths = test_paths[:num_test_images]
    test_subset_features = all_test_features[:num_test_images]

    all_precisions, all_recalls, all_f1s = [], [], []

    for i, query_image_path in tqdm(enumerate(test_subset_paths), total=len(test_subset_paths), desc="Evaluating"):
        query_features = test_subset_features[i].reshape(1, -1)
        similar_images = find_similar_images_knn(query_features, all_test_features, test_paths)

        similar_image_paths = [path for path, _ in similar_images]
        precision, recall, f1 = evaluate_similarity(query_image_path, similar_image_paths, model, transform, device)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)

    print(f"\n--- Overall Performance ---")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")