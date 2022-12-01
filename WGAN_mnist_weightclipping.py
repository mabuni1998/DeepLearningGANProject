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
from torchvision.datasets import MNIST
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
  "epochs": 50,
  "batch_size": 64,
  "latent_dim":100,
  "clipping":0.01
}
config = wandb.config

#Define model
latent_dim = config["latent_dim"]
# The generator takes random `latent` noise and
# turns it into an MNIST image.
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
).to(device)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# The discriminator takes an image (real or fake)
# and decides whether it is generated or not.
discriminator = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=4, stride=2),
    nn.LeakyReLU(0.2),
    nn.Conv2d(64, 128, kernel_size=4, stride=2),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2),
    nn.Conv2d(128, 256, kernel_size=4, stride=2),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2),
    Flatten(),
    nn.Linear(256, 1),
    #nn.Sigmoid()
).to(device)

if __name__ == "__main__":

    run = wandb.init(project="mnist-WGAN_clipping", entity="gan_project_cm",config=config)
    
    
    
    
    # The digit classes to use, these need to be in order because
    classes = np.arange(10)
    
    def one_hot(labels):
        y = torch.eye(len(classes)) 
        return y[labels]
    
    # Define the train and test sets
    #dset_train = MNIST("./", train=True, download=True, transform=Compose([ToTensor()]), target_transform=one_hot)
    #dset_test  = MNIST("./", train=False, transform=Compose([ToTensor()]), target_transform=one_hot)
    
    dset_train = MNIST("./", train=True, download=True, transform=Compose([ToTensor(),Normalize(0.5,0.5)]), target_transform=one_hot)
    dset_test  = MNIST("./", train=False, transform=Compose([ToTensor(),Normalize(0.5,0.5)]), target_transform=one_hot)
    
    
    def stratified_sampler(labels):
        """Sampler that only picks datapoints corresponding to the specified classes"""
        (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
        indices = torch.from_numpy(indices)
        return SubsetRandomSampler(indices)
    
    batch_size = config["batch_size"]
    # The loaders perform the actual work
    train_loader = DataLoader(dset_train, batch_size=batch_size,
                              sampler=stratified_sampler(dset_train.targets), pin_memory=cuda,drop_last=True)
    test_loader  = DataLoader(dset_test, batch_size=batch_size, 
                              sampler=stratified_sampler(dset_test.targets), pin_memory=cuda)
    
    
    
    
    print("Using device:", device)
    
    lr = config["learning_rate"]
    generator_optim = torch.optim.Adam(generator.parameters(), lr, betas=(0.5, 0.999))
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr, betas=(0.5, 0.999))
    #generator_optim = torch.optim.RMSprop(generator.parameters(), lr)
    #discriminator_optim = torch.optim.RMSprop(discriminator.parameters(), lr)
    
    
    
    discriminator_loss, generator_loss = [], []
    
    num_epochs = config["epochs"]
    for epoch in range(num_epochs):
        batch_d_loss, batch_g_loss = [], []
        
        for x,_ in train_loader:
            #Get real data
            x_true = x.to(device)       
            for i in range(5):
              discriminator.zero_grad()
              #Get fake data
              z = torch.randn(batch_size, latent_dim,1,1)
              z.requires_grad=False
              z = z.to(device)
              x_fake = generator(z)
    
              output_true = discriminator(x_true)
              output_fake = discriminator(x_fake.detach())
              
              d_loss = -(output_true  - output_fake).mean()
              d_loss.backward()
              discriminator_optim.step()
              with torch.no_grad():
                  for param in discriminator.parameters():
                      param.clamp_(-config["clipping"], config["clipping"])
        
            output_fake = discriminator(x_fake)
            generator.zero_grad()
            g_loss = -(output_fake).mean()
            g_loss.backward()
            generator_optim.step()
            
            batch_d_loss.append((d_loss).item())
            batch_g_loss.append(g_loss.item())
    
        discriminator_loss.append(np.mean(batch_d_loss))
        generator_loss.append(np.mean(batch_g_loss))
        
    
        # Use generated data
        x_fake.data = x_fake.data
        
        fid_score =calculate_fid(torch.cat([x_fake.data, x_fake.data, x_fake.data], dim=1).to(device),torch.cat([x_true.data,x_true.data, x_true.data], dim=1).to(device))
        print("FID score: {fid:.2f}".format(fid=fid_score))
            
        wandb.log({'epoch': epoch+1, 'Generator loss': g_loss, 'Discriminator loss': d_loss, 'FID Score': fid_score})
    
        wandb.watch(discriminator)
        wandb.watch(generator)
    
    #Save model
    generatorpath = os.getcwd()+'/models/WGAN_clip_mnist_g.pth'
    discriminatorpath = os.getcwd()+'/models/WGAN_clip_mnist_d.pth'
    
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