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
from models_file import get_gan_models
import matplotlib

font = {'family':'serif','size':20}
matplotlib.rc('font',**font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'

#import wandb
#run = wandb.init()
config = {
  "learning_rate": 2e-4,
  "epochs": 100,
  "batch_size": 128,
  "latent_dim":100,
  "clipping":0.02,
  "nch":12,
  "lambda_gp":10
}
generator,d = get_gan_models('gan','mnist',config)

#Load model


#artifact = run.use_artifact('gan_project_cm/mnist-GAN/'+'generator:v21', type='model')
#artifact_dir = artifact.download()
generatorpath = os.getcwd()+'/artifacts/generator-v21/mnist-GAN_g.pth'

"""
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
"""
generator.load_state_dict(torch.load(generatorpath,map_location=device))



# Generate data
with torch.no_grad():
    z = torch.randn(64, 100, 1, 1)
    z = z.to(device)
    z.requires_grad=False
    x_fake = generator(z)

x_fake.data = x_fake.data.cpu()

# -- Plotting --
f, ax = plt.subplots(1, 1, figsize=(9, 9))

ax.set_title('MNIST GAN')
ax.axis('off')

rows, columns = 8, 8
  
canvas = np.zeros((28*rows, columns*28))
for i in range(rows):
    for j in range(columns):
        idx = i % columns + rows * j
        canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_fake.data[idx]
ax.imshow(canvas, cmap='gray')
plt.tight_layout()
cwd=os.getcwd()
plt.savefig(cwd+"/plots/mnist_sample_gan.svg")

