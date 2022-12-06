import matplotlib.pyplot as plt
import numpy as np
plt.style.use(["seaborn-deep", "seaborn-whitegrid"])
import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
#device = 'cpu'
from torch.utils.data import DataLoader
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from functools import reduce
from scipy.linalg import sqrtm
from torch import nn
from scipy.linalg import sqrtm
from torchvision import transforms
import os
from WGAN_weight_cifar10 import generator
import seaborn as sns
sns.set_style("whitegrid")
from torchvision.utils import make_grid
import wandb
from models_file import get_gan_models
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
generator,d = get_gan_models('GP','celeba',config)

#Load model
generatorpath = os.getcwd()+'/models/celeb_WGAN_GP_g.pth'
generator.load_state_dict(torch.load(generatorpath,map_location=device))


def show_image(img):
    img = img.detach().cpu()
    img = img / 2 + 0.5   # unnormalize
    with sns.axes_style("white"):
        plt.figure(figsize=(8, 8))
        plt.imshow(img.permute((1, 2, 0)).numpy())
        plt.axis('off')
        plt.show()


# Generate data
with torch.no_grad():
    z = torch.randn(64, 100, 1, 1)
    z = z.to(device)
    z.requires_grad=False
    x_fake = generator(z)


x_fake.data = x_fake.data.cpu()


show_image(make_grid(x_fake))

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # subtract 0.5 and divide by 0.5
    ]
)
#transform = transforms.ToTensor()
"""
train_set = CIFAR10('./', train=True, download=False,transform=transform)


train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
images, labels = iter(train_loader).next()
show_image(make_grid(images))
"""
"""
# -- Plotting --
f, ax = plt.subplots(1, 1, figsize=(18, 7))

ax.set_title('Samples from generator')
ax.axis('off')

rows, columns = 4, 4
  
canvas = np.zeros((3,32*rows, columns*32))
for i in range(rows):
    for j in range(columns):
        idx = i % columns + rows * j
        canvas[:,i*32:(i+1)*32, j*32:(j+1)*32] = x_fake.data[idx]
ax.imshow(canvas)

cwd=os.getcwd()
plt.savefig(cwd+"/plots/WGAN_weight_cifar10_generator.pdf")
"""