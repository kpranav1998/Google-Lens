import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_extraction import extract_features
from sklearn.neighbors import NearestNeighbors

def evaluate_similarity(query_image_path, database_paths, model, transform, device, top_k=5):
    query_class = 1 if "dog" in query_image_path.lower() else 0
    query_features = extract_features(query_image_path, model, transform, device)
    if query_features is None:
        return 0, 0, 0

    image_features = []
    image_labels = []
    relevant_count = 0

    for db_path in database_paths:
        db_class = 1 if "dog" in db_path.lower() else 0
        if db_class == query_class:
            relevant_count += 1
        features = extract_features(db_path, model, transform, device)
        if features is not None:
            image_features.append(features)
            image_labels.append(db_class)

    if not image_features:
        return 0,0,0

    image_features = np.concatenate(image_features, axis=0)
    similarities = cosine_similarity(query_features, image_features).flatten()
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_labels = [image_labels[i] for i in top_k_indices]

    tp = sum([1 for label in retrieved_labels if label == query_class])
    fp = sum([1 for label in retrieved_labels if label != query_class])
    fn = relevant_count - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def find_similar_images_knn(query_features, all_image_features, database_paths, top_k=5):
    """Finds similar images using pre-computed features and KNN."""
    knn = NearestNeighbors(n_neighbors=top_k, metric='cosine') # Use cosine metric directly
    knn.fit(all_image_features)
    distances, indices = knn.kneighbors(query_features)
    similar_images = [(database_paths[i], 1 - distances.flatten()[j]) for j, i in enumerate(indices.flatten())] # Convert distances to similarities
    return similar_images