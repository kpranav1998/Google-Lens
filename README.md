## Image Similarity Search

### **Objective**

This project explores alternatives to Google Lens by implementing various approaches for image similarity search. We propose and test distinct methods for finding similar images, including:

1. **CNN-based Feature Extraction**  
2. **Autoencoders**  
3. **Siamese Networks**

Each approach is fine-tuned on a cat-and-dog image dataset and compared using metrics like precision, recall, and retrieval accuracy. We consider factors like performance, scalability, and suitability for real-time applications.

### **Dataset**

We utilized a public dataset from Kaggle containing 20,000 images (10,000 cats and 10,000 dogs) accessible through this link: [Download Dataset](https://storage.googleapis.com/kaggle-data-sets/550917/1003830/bundle/archive.zip).

---

### **Approaches Tested**

#### **1. CNN-based Feature Extraction (Pre-trained Models)**

This method leverages pre-trained CNN models (e.g., ResNet, VGG) to extract high-level features from images. These features are then compared using cosine similarity and a K-Nearest Neighbors (KNN) search.

**Steps:**

1. Extract feature embeddings from images using a pre-trained CNN model.
2. Apply K-Nearest Neighbors (KNN) with cosine similarity to retrieve similar images.

**Why KNN with Cosine Similarity:**  
KNN with cosine similarity offers several advantages over using cosine similarity alone. By employing KNN, we retrieve the top *k* nearest neighbors, ensuring that multiple similar images are returned for a given query. This approach improves the robustness of the similarity search by considering the local neighborhood structure in the feature space. Additionally, KNN allows for efficient querying in large datasets by leveraging indexing techniques, making it more scalable than performing pairwise cosine similarity comparisons across the entire dataset.

**References:**
1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet classification with deep convolutional neural networks." In Advances in neural information processing systems.  
2. Simonyan, K., & Zisserman, A. (2014). "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556.

---

#### **2. Autoencoders**

Autoencoders are employed to learn compact representations of images. The encoder part of the network generates embeddings that are compared to measure similarity.

**Steps:**

1. Train an autoencoder on the image dataset to learn image representations.
2. Use the encoder output as feature embeddings.
3. Perform similarity search using KNN with cosine similarity.

**References:**
1. Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). "Extracting and composing robust features with denoising autoencoders." In Proceedings of the 25th international conference on Machine learning.  
2. Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the dimensionality of data with neural networks." Science.

---

#### **3. Siamese Networks**

A Siamese network is designed to learn a distance metric between pairs of images by minimizing a contrastive loss function. This approach allows direct similarity computation without additional feature extraction.

**Steps:**

1. Define a Siamese network with a contrastive loss function.
2. Train the network using pairs of similar and dissimilar images.
3. Use the trained network to compute pairwise distances between images.

**References:**
1. Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., & Shah, R. (1993). "Signature verification using a "Siamese" time delay neural network." In Advances in neural information processing systems.  
2. Koch, G., Zemel, R., & Salakhutdinov, R. (2015). "Siamese neural networks for one-shot image recognition." In ICML Deep Learning Workshop.

---

### **Results**

#### **CNN-based Feature Extraction**

- **Precision\@5:** 0.82  
- **MAP\@5:** 0.9786

#### **Autoencoder**

- **Precision\@5:** 0.8000

#### **Siamese Network**

- **Mean Precision\@5:** 0.4820  
- **MAP\@5:** 0.4940

---

### **Evaluation Metrics**

1. **Precision@k:** Precision at *k* measures the fraction of relevant images among the top *k* retrieved images:
   
   **Precision@k** = (Number of relevant images in top- *k*) / *k*

2. **Mean Average Precision (MAP):** MAP is the mean of the average precision scores for multiple queries. For a single query, Average Precision (AP) is the average of the precision values at different cutoff levels where relevant images are retrieved. MAP is computed as:
   
   **MAP** = (1 / *N*) * Σ (Precision@k for query *i*)
   
   Where *N* is the total number of queries.

---

### **Conclusion**

- **CNN-based Feature Extraction:**  
  - Achieved the best performance in terms of precision and MAP.  
  - It leverages pre-trained models, which are known to capture high-level semantic features, making them highly effective for similarity search.  
  - Computationally efficient for real-time applications due to the use of pre-computed embeddings and KNN-based retrieval.

- **Autoencoders:**  
  - Slightly lower precision compared to CNN-based feature extraction.  
  - More generalized representations might have affected fine-grained similarity detection.  
  - Requires less memory for storing embeddings but is computationally less efficient than pre-trained CNN models.

- **Siamese Networks:**  
  - Performed the worst among the three approaches.  
  - Requires careful pair selection and a larger dataset for effective training.  
  - Computationally intensive during both training and inference, making it less suitable for real-time applications.

- **Scalability Considerations:**  
  - CNN-based feature extraction with KNN indexing scales better for large datasets due to efficient querying mechanisms.  
  - Autoencoders offer a balance between memory efficiency and retrieval accuracy.  
  - Siamese networks may face challenges in scaling due to their pairwise training approach and inference complexity.

### **Future Work**

- Implement additional approaches such as triplet networks or graph-based similarity search.
- Improve computational efficiency by optimizing model inference time.
- Develop a user-friendly GUI or web application for real-time image similarity search.
- Explore data augmentation techniques and better pair selection strategies to improve Siamese network performance.

---

### **How to Run the Project**

1. **Download and unzip the dataset.**

2. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the desired approach script to train the model and test similarity search:**

   - **For Autoencoder approach:**
     ```bash
     python autoencoder.py
     ```
   - **For CNN-based Feature Extraction approach:**
     ```bash
     python cnn.py
     ```
   - **For Siamese Network approach:**
     ```bash
     python siamese.py
     ```

---

### **Dependencies**

- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- numpy
- PIL (Pillow)
- tqdm

---

### **Future Work**

- Implement additional approaches such as triplet networks or graph-based similarity search.
- Improve computational efficiency by optimizing model inference time.
- Develop a user-friendly GUI or web application for real-time image similarity search.

