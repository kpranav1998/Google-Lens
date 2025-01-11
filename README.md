# **Image Similarity Search - Project README**

## **Objective**
The goal of this project is to develop an alternative to Google Lens by implementing **image similarity search** using multiple approaches. The project proposes and tests distinct methods for performing similarity search, including:

1. **CNN-based Feature Extraction**
2. **Autoencoders**
3. **Siamese Networks**

Each approach is fine-tuned on a cat-and-dog image dataset, and their performance is compared using metrics such as precision, recall, and retrieval accuracy. Additionally, computational efficiency and scalability are considered to enable real-time usage scenarios.

## **Dataset**
The dataset consists of **20,000 images**, with **10,000 images** of cats and **10,000 images** of dogs. It is available on Kaggle, and you can download it using [this link](https://storage.googleapis.com/kaggle-data-sets/550917/1003830/bundle/archive.zip).

## **Approaches Tested**

### **1. CNN Feature Extraction (Pre-trained Models)**
In this approach, pre-trained CNN models (e.g., ResNet, VGG) are used to extract high-level features from images. These features are then compared using cosine similarity.

#### **Steps**
- Use a pre-trained CNN model to extract feature embeddings from the images.
- Apply K-Nearest Neighbors (KNN) with cosine similarity to retrieve similar images.

### **2. Autoencoders**
Autoencoders are employed to learn compact representations of images. The encoder part of the network generates embeddings that are compared to measure similarity.

#### **Steps**
- Train an autoencoder on the image dataset to learn image representations.
- Use the encoder output as feature embeddings.
- Perform similarity search using KNN with cosine similarity.

### **3. Siamese Networks**
A Siamese network is designed to learn a distance metric between pairs of images by minimizing contrastive loss. This enables direct similarity computation without additional feature extraction.

#### **Steps**
- Define a Siamese network with a contrastive loss function.
- Train the network using pairs of similar and dissimilar images.
- Use the trained network to compute pairwise distances between images.

## **How to Run the Project**
1. Download and unzip the dataset.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the desired approach script to train the model and test similarity search:
   - For **Autoencoder** approach:
     ```bash
     python autoencoder.py
     ```
   - For **CNN-based Feature Extraction** approach:
     ```bash
     python cnn_feature_extraction.py
     ```
   - For **Siamese Network** approach:
     ```bash
     python siamese.py
     ```

## **Dependencies**
- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- numpy
- PIL (Pillow)
- tqdm

## **Future Work**
- Implement additional approaches such as triplet networks or graph-based similarity search.
- Improve computational efficiency by optimizing model inference time.
- Develop a user-friendly GUI or web application for real-time image similarity search.


