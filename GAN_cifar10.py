import wandb
import matplotlib.pyplot as plt
import numpy as np
plt.style.use(["seaborn-deep", "seaborn-whitegrid"])
import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
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
from FID_score import calculate_fid

wandb.config = {
  "learning_rate": 2e-4,
  "epochs": 3000,
  "batch_size": 64,
  "latent_dim":100,
  "clipping":0.01
}
config = wandb.config

#torch.backends.cudnn.benchmark = True

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

#Define model
latent_dim = config["latent_dim"]
# The generator takes random `latent` noise and
# turns it into an MNIST image.
generator = nn.Sequential(
    # nn.ConvTranspose2d can be seen as the inverse operation
    # of Conv2d, where after convolution we arrive at an upscaled image.
    nn.ConvTranspose2d(latent_dim, 1024, kernel_size=2, stride=2,bias=False),
    nn.BatchNorm2d(1024),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2,bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2,bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2,bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(128, 3, kernel_size=2, stride=2),
    nn.Tanh()
).to(device)

# The discriminator takes an image (real or fake)
# and decides whether it is generated or not.
discriminator = nn.Sequential(
    nn.Conv2d(3, 128, kernel_size=2, stride=2),
    nn.LeakyReLU(0.2),
    nn.Conv2d(128, 256, kernel_size=2, stride=2,bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.4),
    nn.Conv2d(256, 512, kernel_size=2, stride=2,bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.4),
    nn.Conv2d(512, 1024, kernel_size=3, stride=1),
    Flatten(),
    nn.Linear(4096, 1),
    nn.Sigmoid()
).to(device)

if __name__ == "__main__":

    run = wandb.init(project="CIFAR10_GAN", entity="gan_project_cm",config=config)
    
    
    
    
    # The output of torchvision datasets are PIL images in the range [0, 1]. 
    # We transform them to PyTorch tensors and rescale them to be in the range [-1, 1].
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # subtract 0.5 and divide by 0.5
        ]
    )

    batch_size = config["batch_size"]  # both for training and testing

    # Load datasets
    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    # The loaders perform the actual work
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=cuda,drop_last=True,num_workers=1)    
    
    print("Using device:", device)
    
    lr = config["learning_rate"]
    generator_optim = torch.optim.Adam(generator.parameters(), lr, betas=(0.5, 0.999))
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr, betas=(0.5, 0.999))
    #generator_optim = torch.optim.RMSprop(generator.parameters(), lr)
    #discriminator_optim = torch.optim.RMSprop(discriminator.parameters(), lr)
    
    loss_fn = nn.BCELoss()
    
    discriminator_loss, generator_loss = [], []
    
    num_epochs = config["epochs"]
    for epoch in range(num_epochs):
        batch_d_loss, batch_g_loss = [], []
        
        for x,_ in train_loader:
            #Get real data
            x_true = x.to(device)       
            for i in range(1):
              discriminator.zero_grad()
              #Get fake data
              z = torch.randn(batch_size, latent_dim,1,1)
              z.requires_grad=False
              z = z.to(device)
              x_fake = generator(z)
    
              output_true = discriminator(x_true)
              output_fake = discriminator(x_fake.detach())
              
              d_loss = - (1 - output_fake).log().mean() - output_true.log().mean()
              d_loss.backward()
              discriminator_optim.step()
        
            output_fake = discriminator(x_fake)
            generator.zero_grad()
            g_loss = (1 - output_fake).log().mean()
            g_loss.backward()
            generator_optim.step()
            
            batch_d_loss.append((d_loss).item())
            batch_g_loss.append(g_loss.item())
    
        discriminator_loss.append(np.mean(batch_d_loss))
        generator_loss.append(np.mean(batch_g_loss))
        
        if epoch % 5 == 0:
          fid_score =calculate_fid(x_fake.data.to(device),x_true.data.to(device))
          print("FID score: {fid:.2f}".format(fid=fid_score)) 
          wandb.log({'epoch': epoch+1, 'Generator loss': g_loss, 'Discriminator loss': d_loss, 'FID Score': fid_score})
    
        wandb.watch(discriminator)
        wandb.watch(generator)
    
    #Save model
    generatorpath = os.getcwd()+'/models/GAN_CIFAR10_g.pth'
    discriminatorpath = os.getcwd()+'/models/GAN_CIFAR10_d.pth'
    
    torch.save(generator.state_dict(),generatorpath )
    torch.save(discriminator.state_dict(),discriminatorpath )
    # Save as artifact for version control.
    artifact = wandb.Artifact('discriminator', type='model')
    artifact.add_file(discriminatorpath)
    run.log_artifact(artifact)
    artifact = wandb.Artifact('generator', type='model')
    artifact.add_file(generatorpath)
    run.log_artifact(artifact)
    wandb.run.finish()

