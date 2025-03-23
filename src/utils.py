import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def load_model(model_cls, path, device):
    """
    Load a saved model
    """
    model = model_cls()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, dataset_name):
    """
    Preprocess an image for prediction with improved preprocessing for real-world images
    """
    if dataset_name == "CIFAR-10":
        # CIFAR-10 preprocessing
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        img = Image.open(image_path).convert('RGB')
    
    elif dataset_name == "Fashion-MNIST":
        # Fashion-MNIST preprocessing with enhanced real-world image handling
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.Resize((28, 28)),
            # Add center crop to handle different aspect ratios better
            transforms.CenterCrop((28, 28)),
            # Add additional preprocessing to make images more similar to Fashion-MNIST
            transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, 1.5)),  # Enhance contrast
            transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 0.8)),  # Slightly darken
            transforms.ToTensor(),
            # Normalize with Fashion-MNIST mean and std
            transforms.Normalize((0.2860,), (0.3530,)),
            # Add threshold to create more binary-like images like Fashion-MNIST
            transforms.Lambda(lambda x: (x > 0.1).float())
        ])
        img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return transform(img).unsqueeze(0)  # Add batch dimension

def predict_image(model, image_tensor, device, classes):
    """
    Make a prediction on an image
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
    
    return {
        'class_idx': predicted.item(),
        'class_name': classes[predicted.item()],
        'confidence': conf.item(),
        'all_probs': probs.squeeze().cpu().numpy()
    }

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, classes):
    """
    Plot confusion matrix
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig

def plot_prediction_bar(prediction, classes):
    """
    Plot prediction probabilities as a bar chart
    """
    probs = prediction['all_probs']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Probability')
    ax.set_title('Class Probabilities')
    
    # Highlight the predicted class
    predicted_class = prediction['class_idx']
    ax.get_children()[predicted_class].set_color('green')
    
    plt.tight_layout()
    return fig