# PixelRNN Image Completion

A deep learning project that uses a PixelRNN-inspired U-Net architecture to reconstruct occluded (masked) images. The model learns to fill in missing regions of images through a combination of pixel-level and perceptual loss functions.

## 🚀 Features

- **PixelRNN-inspired U-Net Architecture**: Modified U-Net with PixelRNN concepts for sequential image reconstruction
- **Perceptual Loss**: Uses VGG16 features for better visual quality
- **Interactive Web Interface**: Streamlit-based UI for easy image upload and reconstruction
- **Real-time Inference**: Fast image completion with GPU acceleration
- **Model Checkpointing**: Automatic saving of best models during training

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/MuhammadMaaz7/pixelrnn-image-completion.git
cd pixelrnn-image-completion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Structure

The project expects the following dataset structure:
```
dataset_A2/
├── train/
│   ├── occluded_images/    # Training images with occlusions
│   └── original_images/    # Corresponding original images
└── occluded_test/          # Test images with occlusions
```

## 🤖 Model Files

**Important**: Model files (`.pth`) are not included in the repository due to their large size. You have two options:

### Option 1: Train Your Own Model
```bash
python pixelrnn_train.py
```
This will create `outputs/pixelrnn_best_model.pth` after training.

### Option 2: Use Pre-trained Model
If you have a pre-trained model:
1. Ensure the `outputs/` directory exists
2. Place your model file as `outputs/pixelrnn_best_model.pth`

**Note**: The Streamlit app will show an error if no trained model is found. Train the model first or ensure you have a valid checkpoint in the `outputs/` directory.

## 🏃‍♂️ Quick Start

### Training the Model

```bash
python pixelrnn_train.py
```

The training script will:
- Load the dataset from `dataset_A2/`
- Train the PixelRNN-inspired U-Net model
- Save checkpoints to `outputs/`
- Use early stopping to prevent overfitting

### Running the Web Interface

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` to use the interactive interface.

## 🏗️ Model Architecture

The model uses a U-Net architecture with the following components:

- **Encoder**: Convolutional blocks with max pooling for feature extraction
- **Decoder**: Upsampling blocks with skip connections for reconstruction
- **Loss Function**: Combination of MSE loss and perceptual loss using VGG16 features

### Key Features:
- Batch normalization for stable training
- Dropout for regularization
- Skip connections for preserving fine details
- Perceptual loss for better visual quality

## 📈 Training Configuration

Default training parameters:
- **Image Size**: 128x128
- **Batch Size**: 4
- **Learning Rate**: 1e-4
- **Epochs**: 25
- **Early Stopping**: 5 epochs patience

## 🎯 Usage Examples

### Training from Scratch
```python
from pixelrnn_train import train_pixelrnn
model, val_loader = train_pixelrnn()
```

### Loading Pre-trained Model
```python
import torch
from pixelrnn_train import PixelRNNishUNet

model = PixelRNNishUNet()
checkpoint = torch.load('outputs/pixelrnn_best_model.pth')
model.load_state_dict(checkpoint['model_state'])
```

### Inference on Single Image
```python
from PIL import Image
from torchvision import transforms

# Load and preprocess image
image = Image.open('path/to/occluded_image.jpg')
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
input_tensor = transform(image).unsqueeze(0)

# Generate reconstruction
with torch.no_grad():
    output = model(input_tensor)
    reconstructed = transforms.ToPILImage()(output.squeeze())
```

## 📁 Project Structure

```
pixelrnn-image-completion/
├── app.py                  # Streamlit web interface
├── pixelrnn_train.py      # Training script and model definition
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── LICENSE               # MIT License
├── .gitignore           # Git ignore rules
├── dataset_A2/          # Dataset directory
│   ├── train/
│   │   ├── occluded_images/
│   │   └── original_images/
│   └── occluded_test/
└── outputs/             # Model checkpoints and outputs
```

## 🔧 Configuration

You can modify training parameters in `pixelrnn_train.py`:

```python
IMAGE_SIZE = 128          # Input image size
BATCH_SIZE = 4           # Training batch size
EPOCHS = 25              # Maximum training epochs
LR = 1e-4               # Learning rate
EARLY_STOPPING_PATIENCE = 5  # Early stopping patience
```

## 📊 Model Performance

The model uses two loss components:
- **Pixel Loss (MSE)**: Ensures pixel-level accuracy
- **Perceptual Loss**: Maintains visual quality using VGG16 features

Training includes:
- Learning rate scheduling
- Gradient clipping
- Early stopping
- Model checkpointing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- U-Net architecture inspiration from [Ronneberger et al.](https://arxiv.org/abs/1505.04597)
- PixelRNN concepts from [van den Oord et al.](https://arxiv.org/abs/1601.06759)
- Perceptual loss implementation using VGG16 features

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.