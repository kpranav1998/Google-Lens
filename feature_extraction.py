import torch
from PIL import Image

def extract_features(image_path, model, transform, device):
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
        if hasattr(model, 'encoder'):  # Check for autoencoder
            features = model.encoder(image_tensor)
        else:
            features = model(image_tensor) # Regular CNN
        features = torch.flatten(features, 1)
    return features.cpu().numpy()