import matplotlib.pyplot as plt
import numpy as np
plt.style.use(["seaborn-deep", "seaborn-whitegrid"])
import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
from torch.utils.data import DataLoader
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from functools import reduce
from scipy.linalg import sqrtm
from torch import nn
from scipy.linalg import sqrtm
from torchvision import transforms
import os
from FID_score import calculate_fid

import wandb
run = wandb.init()

#Save model
generatorpath = os.getcwd()+'/models/GAN_mnist_g.pth'
discriminatorpath = os.getcwd()+'/models/GAN_mnist_d.pth'


#artifact = run.use_artifact('gan_project_cm/mnist-GAN/'+'generator:v0', type='model')
#artifact_dir = artifact.download()

latent_dim =100
generator = nn.Sequential(
    # nn.ConvTranspose2d can be seen as the inverse operation
    # of Conv2d, where after convolution we arrive at an
    # upscaled image.
    nn.ConvTranspose2d(latent_dim, 256, kernel_size=3, stride=2),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
    #nn.Sigmoid() # Image intensities are in [0, 1]
    nn.Tanh()
)
generator.load_state_dict(torch.load(generatorpath,map_location=device))



# Generate data
with torch.no_grad():
    z = torch.randn(64, 100, 1, 1)
    z = z.to(device)
    z.requires_grad=False
    x_fake = generator(z)

x_fake.data = x_fake.data.cpu()

# -- Plotting --
f, ax = plt.subplots(1, 1, figsize=(18, 7))

ax.set_title('Samples from generator')
ax.axis('off')

rows, columns = 8, 8
  
canvas = np.zeros((28*rows, columns*28))
for i in range(rows):
    for j in range(columns):
        idx = i % columns + rows * j
        canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_fake.data[idx]
ax.imshow(canvas, cmap='gray')

cwd=os.getcwd()
plt.savefig(cwd+"plots/GAN_mnist_generator.pdf")

