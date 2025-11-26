import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ------------------ Configuration ------------------
DATA_ROOT = "dataset_A2"  # Relative path to dataset
SAVE_DIR = "outputs"
MODEL_FILENAME = "pixelrnn_best_model.pth"
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-4
EARLY_STOPPING_PATIENCE = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------ Dataset Classes ------------------
class OccludedDataset(Dataset):
    """Dataset for training: includes occluded and original image pairs."""
    def __init__(self, root, split="train"):
        self.masked_dir = os.path.join(root, split, "occluded_images")
        self.original_dir = os.path.join(root, split, "original_images")
        self.masked_imgs = sorted(os.listdir(self.masked_dir))
        self.original_imgs = sorted(os.listdir(self.original_dir))
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.masked_imgs)

    def __getitem__(self, idx):
        masked = Image.open(os.path.join(self.masked_dir, self.masked_imgs[idx])).convert("RGB")
        original = Image.open(os.path.join(self.original_dir, self.original_imgs[idx])).convert("RGB")
        return self.transform(masked), self.transform(original)


class TestDataset(Dataset):
    """Dataset for testing on occluded images only."""
    def __init__(self, root):
        self.root = root
        self.imgs = sorted(os.listdir(root))
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.imgs[idx])).convert("RGB")
        return self.transform(img), self.imgs[idx]


# ------------------ Model Definition ------------------
class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and dropout."""
    def __init__(self, in_c, out_c, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.conv(x)


class PixelRNNishUNet(nn.Module):
    """Modified U-Net inspired by PixelRNN for image reconstruction."""
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.center = ConvBlock(256, 512)

        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        self.final = nn.Conv2d(64, 3, 1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        c = self.center(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up(c), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        out = torch.sigmoid(self.final(d1))
        return out


# ------------------ Perceptual Loss ------------------
class PerceptualLoss(nn.Module):
    """Computes feature-level similarity using pretrained VGG16."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:9].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(self.vgg(pred), self.vgg(target))


# ------------------ Evaluation Function ------------------
def evaluate_and_visualize(model, loader, num_images=5):
    """Display comparison between occluded and reconstructed outputs."""
    model.eval()
    with torch.no_grad():
        for i, (masked, names) in enumerate(loader):
            masked = masked.to(device)
            output = model(masked).clamp(0, 1)

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(np.transpose(masked[0].cpu(), (1, 2, 0)))
            plt.title("Occluded Input")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(np.transpose(output[0].cpu(), (1, 2, 0)))
            plt.title("Reconstructed Output")
            plt.axis("off")

            plt.suptitle(f"File: {names[0]}")
            plt.show()

            if i >= num_images - 1:
                break


# ------------------ Training Function ------------------
def train_pixelrnn():
    """Main training loop with checkpointing and early stopping."""
    train_dataset = OccludedDataset(DATA_ROOT, "train")
    val_dataset = TestDataset(os.path.join(DATA_ROOT, "occluded_test"))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = PixelRNNishUNet().to(device)
    mse_loss = nn.MSELoss()
    perceptual_loss = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for masked, original in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            masked, original = masked.to(device), original.to(device)
            optimizer.zero_grad()

            output = model(masked)
            loss_pixel = mse_loss(output, original)
            loss_perceptual = perceptual_loss(output, original)
            loss = loss_pixel + 0.1 * loss_perceptual

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

        # Save model checkpoints
        ckpt = {"model_state": model.state_dict(), "val_loss": avg_loss, "epoch": epoch + 1}
        torch.save(ckpt, os.path.join(SAVE_DIR, f"pixelrnn_epoch_{epoch+1}.pth"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(ckpt, os.path.join(SAVE_DIR, "pixelrnn_best_model.pth"))
            print("Saved new best model.")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping: no further improvement.")
            break

    return model, val_loader


# ------------------ Main Execution ------------------
if __name__ == "__main__":
    ckpt_path = os.path.join(SAVE_DIR, "pixelrnn_best_model.pth")

    if os.path.exists(ckpt_path):
        print("Found checkpoint. Loading model for evaluation...")
        model = PixelRNNishUNet().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded model (epoch {ckpt['epoch']}) | val_loss={ckpt['val_loss']:.4f}")

        val_loader = DataLoader(TestDataset(os.path.join(DATA_ROOT, "occluded_test")),
                                batch_size=1, shuffle=False, num_workers=0)
        evaluate_and_visualize(model, val_loader, num_images=5)
    else:
        print("No checkpoint found. Starting new training session...")
        model, val_loader = train_pixelrnn()
        evaluate_and_visualize(model, val_loader)