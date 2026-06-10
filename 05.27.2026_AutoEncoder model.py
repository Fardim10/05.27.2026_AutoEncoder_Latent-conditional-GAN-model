import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import re
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# ==================== Device ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==================== Paths ====================
root_data_dirs = [
    r"D:\Simu fact_welding\Shapes\Rectangle\Simulation images\Temperature_ASTM color\Temp_Training images\Temp_100A_3mms",
    r"D:\Simu fact_welding\Shapes\Rectangle\Simulation images\Temperature_ASTM color\Temp_Training images\Temp_100A_7mms",
    r"D:\Simu fact_welding\Shapes\Rectangle\Simulation images\Temperature_ASTM color\Temp_Training images\Temp_100A_12mms"]
output_dir = r"D:\Deep learning Model\Training\Temp_100A_3mms_Color_Single folder1"
os.makedirs(output_dir, exist_ok=True)
model_save_dir = r"D:\Deep learning Model\Training\Temp_100A_3mms_Color_Single folder1"
os.makedirs(model_save_dir, exist_ok=True)

# ==================== Hyperparameters ====================
image_size = 64
batch_size = 8
num_epochs_ae = 400
num_epochs_gen = 400
lr = 0.0002
betas = (0.5, 0.999)
cond_dim = 3
noise_dim = 100
latent_dim = 256
base_channels = 64
save_every = 50
lambda_recon = 10.0 

# ==================== Transform ====================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# ==================== Dataset ====================
class MeltPoolDataset(Dataset):
    def __init__(self, root_data_dirs, transform=None, max_frames_per_condition=150):
        self.samples = []
        self.transform = transform
        if isinstance(root_data_dirs, str):
            root_data_dirs = [root_data_dirs]
        
        for folder_path in root_data_dirs:
            folder_name = os.path.basename(folder_path)
            match = re.search(r"Temp_(\d+)A_(\d+)mms", folder_name)
            if not match:
                raise ValueError(f"Folder name does not match pattern 'Temp_XXA_XXmms': {folder_name}")
            current_A = float(match.group(1))
            speed_mms = float(match.group(2))
            
            image_files = sorted(glob.glob(os.path.join(folder_path, "*")))
            image_files = [f for f in image_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            n = min(len(image_files), max_frames_per_condition)
            if n == 0:
                raise ValueError(f"No image files found in {folder_path}")
            
            for idx, img_path in enumerate(image_files[:n]):
                timestep = idx / (n - 1) if n > 1 else 0.0
                self.samples.append((img_path, current_A, speed_mms, timestep))
            
            print(f"Loaded {n} samples from {folder_name} (Current={current_A}A, Speed={speed_mms} mm/s)")
        
        print(f"Total samples loaded: {len(self.samples)} (should be ~450)")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, current_A, speed_mms, timestep = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        cond = torch.tensor([
            current_A / 250.0,
            speed_mms / 12.0,
            timestep
        ], dtype=torch.float32)
        return image, cond

dataset = MeltPoolDataset(root_data_dirs=root_data_dirs, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available())
# ==================== AutoEncoder ====================
class Encoder(nn.Module):
    def __init__(self, base_channels=64, latent_dim=256):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten())
        self.fc = nn.Linear(base_channels * 8 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.main(x)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, base_channels=64, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 4 * 4)
        self.main = nn.Sequential(
            nn.Unflatten(1, (base_channels * 8, 4, 4)),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 3, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, z):
        x = self.fc(z)
        return self.main(x)

class AutoEncoder(nn.Module):
    def __init__(self, base_channels=64, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(base_channels, latent_dim)
        self.decoder = Decoder(base_channels, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

ae = AutoEncoder(base_channels=base_channels, latent_dim=latent_dim).to(device)
optimizer_ae = optim.Adam(ae.parameters(), lr=lr, betas=betas)
criterion = nn.MSELoss()

for epoch in range(num_epochs_ae):
    ae.train()
    epoch_loss = 0.0
    for images, _ in tqdm(dataloader, desc=f"AE Epoch {epoch+1}/{num_epochs_ae}"):
        images = images.to(device)
        optimizer_ae.zero_grad()
        recon = ae(images)
        loss = criterion(recon, images)
        loss.backward()
        optimizer_ae.step()
        epoch_loss += loss.item()
    print(f"AE Epoch [{epoch+1}/{num_epochs_ae}] Loss: {epoch_loss / len(dataloader):.6f}")

torch.save(ae.state_dict(), os.path.join(model_save_dir, "autoencoder.pth"))

ae.eval()
with torch.no_grad():
    test_images, _ = next(iter(dataloader))
    test_images = test_images.to(device)
    recon_images = ae(test_images)
    mse = torch.mean((test_images - recon_images) ** 2)
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse + 1e-8))
    print(f"AE Reconstruction PSNR (Speed=3 batch): {psnr.item():.2f} dB → {'real' if psnr.item() > 22.0 else 'Fake'}")

# ==================== Latent Conditional GAN (Generator + Discriminator) ====================
class LatentGenerator(nn.Module):
    def __init__(self, noise_dim=100, cond_dim=3, latent_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, latent_dim))
    
    def forward(self, noise, cond):
        x = torch.cat([noise, cond], dim=1)
        return self.model(x)

class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim=256, cond_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1))
    
    def forward(self, latent, cond):
        x = torch.cat([latent, cond], dim=1)
        return self.model(x)

generator = LatentGenerator(noise_dim, cond_dim, latent_dim).to(device)
discriminator = LatentDiscriminator(latent_dim, cond_dim).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr*0.5, betas=betas)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr*0.5, betas=betas)
adversarial_loss = nn.BCEWithLogitsLoss()

d_losses = []
g_losses = []          
total_losses = []      

ae.eval()  

for epoch in range(num_epochs_gen):
    generator.train()
    discriminator.train()
    
    epoch_d_loss = 0.0
    epoch_g_adv_loss = 0.0
    epoch_total_loss = 0.0
    
    for images, conds in tqdm(dataloader, desc=f"GAN Epoch {epoch+1}/{num_epochs_gen}"):
        images = images.to(device)
        conds = conds.to(device)
        batch_size_cur = images.size(0)
        
        # Latent from AutoEncoder
        real_latent = ae.encode(images)
        
        # Random noise
        noise = torch.rand(batch_size_cur, noise_dim, device=device)
        
        # ==================== Train Discriminator ====================
        fake_latent = generator(noise, conds).detach()
        
        d_real = discriminator(real_latent, conds)
        d_fake = discriminator(fake_latent, conds)
        
        d_loss_real = adversarial_loss(d_real, torch.ones_like(d_real))
        d_loss_fake = adversarial_loss(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # ==================== Train Generator ====================
        fake_latent = generator(noise, conds)
        d_fake = discriminator(fake_latent, conds)
        
        g_adv_loss = adversarial_loss(d_fake, torch.ones_like(d_fake))
        
        # Reconstruction loss
        gen_images = ae.decode(fake_latent)
        recon_loss = criterion(gen_images, images)
        
        total_g_loss = g_adv_loss + lambda_recon * recon_loss
        
        optimizer_G.zero_grad()
        total_g_loss.backward()
        optimizer_G.step()
        
        # Total losses
        epoch_d_loss += d_loss.item()
        epoch_g_adv_loss += g_adv_loss.item()
        epoch_total_loss += total_g_loss.item()
    
    #  Averages epochs
    avg_d = epoch_d_loss / len(dataloader)
    avg_g_adv = epoch_g_adv_loss / len(dataloader)
    avg_total = epoch_total_loss / len(dataloader)
    
    d_losses.append(avg_d)
    g_losses.append(avg_g_adv)
    total_losses.append(avg_total)
    
    print(f"GAN Epoch [{epoch+1}/{num_epochs_gen}] "
          f"D_loss: {avg_d:.6f} | G_adv_loss: {avg_g_adv:.6f} | Total_G_loss: {avg_total:.6f}")
    
    # 85% accuracy heuristic 
    if avg_total < 0.015:
        print("Generator reached 85% accuracy threshold → stopping early")
        break
    
    if (epoch + 1) % save_every == 0:
        torch.save(generator.state_dict(), os.path.join(model_save_dir, f"generator_epoch_{epoch+1}.pth"))

# Save final generator
torch.save(generator.state_dict(), os.path.join(model_save_dir, "generator.pth"))
print("Generator model saved to E:\\generator.pth")

# ==================== Plot Losses (Generator, Discriminator, Total) ====================
plt.figure(figsize=(12, 6))
plt.plot(d_losses, label="Discriminator Loss", color="blue")
plt.plot(g_losses, label="Generator Adversarial Loss", color="orange")
plt.plot(total_losses, label="Total Generator Loss", color="red", linestyle="--")
plt.title("Conditional GAN Training Losses (Latent Space)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = os.path.join(model_save_dir, "loss_plot_GAN.png")
plt.savefig(loss_plot_path)
plt.show()
print(f"Loss plot saved to: {loss_plot_path}")