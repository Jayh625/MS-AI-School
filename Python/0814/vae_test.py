import torch
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from vae_model import VAE

latent_dim = 20
model = VAE()
model.load_state_dict(torch.load("./vae_model.pth", map_location='cpu'))
model.eval()

with torch.no_grad() :
    test_samples = torch.randn(64, latent_dim)
    generated_samples = model.decoder(test_samples).view(64,1,28,28)
    save_image(generated_samples[0], 'generated_sample_one.png')
    save_image(generated_samples, 'generated_sample.png')

fig, axes = plt.subplots(8, 8, figsize=(10,10))
for i, ax in enumerate(axes.flat) :
    ax.imshow(generated_samples[i][0], cmap='gray')
    ax.axis('off')
plt.show()