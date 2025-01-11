from sklearn.neighbors import NearestNeighbors

def find_similar_images_knn(query_features, all_image_features, database_paths, top_k=5):
    knn = NearestNeighbors(n_neighbors=top_k, metric='cosine')
    knn.fit(all_image_features)
    distances, indices = knn.kneighbors(query_features)
    similar_images = [(database_paths[i], 1 - distances.flatten()[j]) for j, i in enumerate(indices.flatten())]
    return similar_images