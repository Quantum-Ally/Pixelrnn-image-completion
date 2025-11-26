import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = "dataset_A2"
SAVE_DIR = "outputs_new"
MODEL_FILENAME = "pixelrnn_best_model.pth"
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE = 64
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
EARLY_STOPPING_PATIENCE = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

# PixelRNN Core: RowLSTM block
class RowLSTM(nn.Module):
    """RowLSTM used in PixelRNN to model pixel dependencies across rows.

    This implementation processes the feature map row-by-row (height dimension).
    Shapes:
      x: (B, C, H, W)
      returns: (B, hidden_dim, H, W)
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_conv = nn.Conv2d(input_dim, 4 * hidden_dim, kernel_size=1)
        self.hidden_conv = nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        B, C, H, W = x.shape

        h_t = torch.zeros(B, self.hidden_dim, 1, W, device=x.device)
        c_t = torch.zeros(B, self.hidden_dim, 1, W, device=x.device)

        outputs = []
        for i in range(H):
            x_t = x[:, :, i, :].unsqueeze(2)

            gates = self.input_conv(x_t) + self.hidden_conv(h_t)

            i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=1)

            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            o_gate = torch.sigmoid(o_gate)
            g_gate = torch.tanh(g_gate)

            c_t = f_gate * c_t + i_gate * g_gate
            h_t = o_gate * torch.tanh(c_t)

            outputs.append(h_t)

        return torch.cat(outputs, dim=2)

# PixelRNN Model
class PixelRNN(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=128, n_layers=2):
        super().__init__()

        self.input_conv = nn.Conv2d(input_channels, hidden_dim, kernel_size=7, padding=3)

        self.rnn_layers = nn.ModuleList([RowLSTM(hidden_dim, hidden_dim) for _ in range(n_layers)])

        self.output_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.input_conv(x)
        for rnn in self.rnn_layers:
            out = rnn(out)
        out = self.output_conv(out)
        return out


# Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:9].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(self.vgg(pred), self.vgg(target))

# Visualization
def visualize_results(model, loader, num_images=5):
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

# Training Loop
def train_pixelrnn():
    train_dataset = OccludedDataset(DATA_ROOT, "train")
    val_dataset = TestDataset(os.path.join(DATA_ROOT, "occluded_test"))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = PixelRNN().to(device)
    mse_loss = nn.MSELoss()
    perceptual_loss = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2, factor=0.5, verbose=True)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for masked, original in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            masked, original = masked.to(device), original.to(device)
            optimizer.zero_grad()

            output = model(masked)
            loss_pixel = mse_loss(output, original)
            loss_perc = perceptual_loss(output, original)
            loss = loss_pixel + 0.1 * loss_perc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

        ckpt = {"model_state": model.state_dict(), "val_loss": avg_loss, "epoch": epoch + 1}
        torch.save(ckpt, os.path.join(SAVE_DIR, f"pixelrnn_epoch_{epoch+1}.pth"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(ckpt, os.path.join(SAVE_DIR, MODEL_FILENAME))
            print("✅ Saved new best model.")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("⛔ Early stopping.")
            break

    return model, val_loader

if __name__ == "__main__":
    ckpt_path = os.path.join(SAVE_DIR, MODEL_FILENAME)

    if os.path.exists(ckpt_path):
        print("Found checkpoint. Loading model...")
        model = PixelRNN().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded model from epoch {ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}")
        val_loader = DataLoader(TestDataset(os.path.join(DATA_ROOT, "occluded_test")),
                                batch_size=1, shuffle=False, num_workers=0)
        visualize_results(model, val_loader)
    else:
        print("No checkpoint found. Starting new training session...")
        model, val_loader = train_pixelrnn()
        visualize_results(model, val_loader)
