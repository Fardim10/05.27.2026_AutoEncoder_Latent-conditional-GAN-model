import torch
import torch.nn as nn
from torchvision import utils as vutils
import os

# ==================== Device ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== Paths ====================
model_save_dir = r"D:\Deep learning Model\Training\Temp_100A_3mms_Color_Single folder1"
output_dir = r"D:\Deep learning Model\Training\Temp_100A_3mms_Color_Single folder1"
os.makedirs(output_dir, exist_ok=True)

# ==================== Hyperparameters ====================
noise_dim = 100
cond_dim = 3
latent_dim = 256
base_channels = 64

# ==================== Model Definitions (FULL AutoEncoder + Generator) ====================
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
            nn.Flatten()
        )
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
            nn.Tanh()
        )

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
            nn.Linear(1024, latent_dim)
        )
    
    def forward(self, noise, cond):
        x = torch.cat([noise, cond], dim=1)
        return self.model(x)


# ==================== Load Trained Models ====================

ae = AutoEncoder(base_channels=base_channels, latent_dim=latent_dim).to(device)
ae.load_state_dict(
    torch.load(os.path.join(model_save_dir, "autoencoder.pth"),
               map_location=device, weights_only=True)
)
ae.eval()
print("✓ AutoEncoder (decoder) loaded successfully")

# Load Generator
generator = LatentGenerator(noise_dim=noise_dim, cond_dim=cond_dim, latent_dim=latent_dim).to(device)
generator.load_state_dict(
    torch.load(os.path.join(model_save_dir, "generator.pth"),
               map_location=device, weights_only=True)
)
generator.eval()
print("✓ Generator loaded successfully")

# ==================== Inference Function ====================
def generate_images(current_A=100.0, speed_mms=5.0, num_images=150, save_folder=None):
    """
    Generate 150 melt-pool images for ANY current and speed (known or unknown).
    """
    if save_folder is None:
        save_folder = os.path.join(output_dir, f"generated_{int(current_A)}A_{speed_mms:.1f}mms")
    os.makedirs(save_folder, exist_ok=True)
    
    current_norm = current_A / 250.0
    speed_norm = speed_mms / 12.0
    
    with torch.no_grad():
        for i in range(num_images):
            timestep = i / (num_images - 1) if num_images > 1 else 0.0
            
            cond = torch.tensor([[current_norm, speed_norm, timestep]],
                                dtype=torch.float32).to(device)
            
            noise = torch.rand(1, noise_dim, device=device)
            
            # Generate latent → image
            gen_latent = generator(noise, cond)
            gen_image = ae.decode(gen_latent) 
            
            # Convert back to [0,1] for saving
            gen_image = (gen_image * 0.5 + 0.5).clamp(0, 1)
            
            # Save frame
            vutils.save_image(gen_image, os.path.join(save_folder, f"generated_frame_{i:04d}.png"))
    
    print(f" Successfully generated {num_images} images for "
          f"Current={current_A}A, Speed={speed_mms} mm/s")
    print(f"   Saved to: {save_folder}")

# ==================== Image Generation ====================
if __name__ == "__main__":
    
    generate_images(current_A=100.0, speed_mms=3.0, num_images=150,
                    save_folder=os.path.join(output_dir, "generated_100A_3mms"))

    print("\n2. Generating for UNKNOWN parameters (example: Speed=5 mm/s) ...")
    generate_images(current_A=100.0, speed_mms=5.0, num_images=150)
