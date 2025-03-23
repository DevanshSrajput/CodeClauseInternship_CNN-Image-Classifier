# 🧠 CNN Image Classifier: Teaching Computers To See Stuff

> "Is that a bird? Is that a plane? No, it's my CNN telling me it's actually a 'Pullover' with 92% confidence!" 

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.7+-brightgreen)
![PyTorch](https://img.shields.io/badge/pytorch-1.9+-orange)
![Fashion Sense](https://img.shields.io/badge/fashion%20sense-impeccable-pink)

## 👀 What's This All About?

Ever wanted to build your own AI that can tell the difference between a sneaker and a sandal? Or maybe distinguish between an airplane and a frog? Well, look no further! This image classifier app uses the magic of Convolutional Neural Networks (CNNs) to identify what's in an image with impressive accuracy - and it does so with a slick UI that won't make your eyes bleed!

Built with PyTorch for the brain power and PyQt5 for the pretty face, this app lets you:

- 🎯 Train models on CIFAR-10 (colorful things like animals and vehicles) or Fashion-MNIST (clothing items for the stylish AI)
- 📊 Watch your model learn in real-time with fancy graphs
- 📸 Upload your own images and let the AI guess what they are
- 🧪 Fiddle with preprocessing options to make your images more recognizable
- 💾 Save your brilliant models for later use

## 🚀 Getting Started

### Prerequisites

Before diving in, make sure you've got:
- Python 3.7+ (because we're not savages)
- A computer with a pulse (GPU recommended but not required)
- Basic understanding of what a neural network is (or willingness to pretend you do)

### Installation

```bash
# Clone this repo like you mean it
git clone https://github.com/yourusername/cnn-image-classifier.git

# Enter the matrix
cd cnn-image-classifier

# Install the dependencies (it's like feeding your AI baby)
pip install -r requirements.txt

# Run the app and feel the power!
python main.py
```

## 🎮 How to Use

### Training Mode

1. Select a dataset (CIFAR-10 for variety, Fashion-MNIST for speed)
2. Set your training parameters:
   - Epochs: 5-10 is usually enough (patience is a virtue, but who's got time?)
   - Batch Size: Bigger = faster but hungrier (like me at an all-you-can-eat buffet)
   - Learning Rate: 0.003 is a good start (not too eager, not too lazy)
3. Hit "Train Model" and watch the magic happen!
4. Marvel at the real-time graphs as your model gets smarter before your very eyes

### Prediction Mode

1. Load your freshly baked model (or use a pre-trained one, we won't judge)
2. Upload an image of something cool
3. Adjust preprocessing options if your image is being stubborn
4. Hit "Predict" and prepare to be amazed (or amused)
5. Check out the confidence scores - is your AI confident or having an existential crisis?

### 🧠 The Technical Bits

CNN Architecture:
┌─────────────────────┐
│     Input Image     │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Conv Layer 1 + BN  │ 32 filters, 3x3
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│      MaxPool 2x2    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Conv Layer 2 + BN  │ 64 filters, 3x3
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│      MaxPool 2x2    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Conv Layer 3 + BN  │ 128 filters, 3x3
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│      MaxPool 2x2    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│      Dropout 25%    │ (To prevent overfitting... and overconfidence)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Fully Connected   │ 512/256 neurons
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│      Dropout 25%    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│    Output (10)      │
└─────────────────────┘

## 📊 Datasets

- **CIFAR-10**: 60,000 32x32 color images across 10 classes
  - Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
  - "When you absolutely need to know if that blob is a dog or a cat"

- **Fashion-MNIST**: 70,000 28x28 grayscale images across 10 classes
  - Classes: t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot
  - "For when your AI wants to give you fashion advice"

## 💡 Pro Tips

- Fashion-MNIST trains faster and is perfect for testing
- Use batch size 128 for quick results if your GPU can handle it
- If your model keeps guessing "Bag" for everything, try the preprocessing options:
  - Enhance contrast for better feature recognition
  - Apply thresholding to create more binary-like images (just like in the dataset)
  - Sometimes inverting the colors helps (Fashion-MNIST has white objects on black backgrounds)

## 🤔 Why Did I Make This?

Because teaching computers to see is almost as fun as teaching humans to code! Plus, everyone needs an AI that can tell them they're wearing a pullover when they thought it was clearly a t-shirt.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- The PyTorch team for making neural networks (almost) painless
- Caffeine, for obvious reasons
- You, for actually reading this far into the README!

#### Made with ❤️ and a healthy dose of confusion about whether that's actually a shirt or a pullover.

