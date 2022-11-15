import wandb
wandb.init(project="mnist-GAN", entity="gan_project_cm")
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
import os

wandb.config = {
  "learning_rate": 2e-4,
  "epochs": 50,
  "batch_size": 64,
  "latent_dim":100
}
config = wandb.config



def calculate_fid(train, target):
	# calculate mean and covariance statistics
	mu1, sigma1 = train.mean(axis=0), np.cov(train, rowvar=False)
	mu2, sigma2 = target.mean(axis=0), np.cov(target, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


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
                          sampler=stratified_sampler(dset_train.train_labels), pin_memory=cuda,drop_last=True)
test_loader  = DataLoader(dset_test, batch_size=batch_size, 
                          sampler=stratified_sampler(dset_test.test_labels), pin_memory=cuda)



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
    nn.Sigmoid()
).to(device)

print("Using device:", device)

lr = config["learning_rate"]
generator_optim = torch.optim.Adam(generator.parameters(), lr, betas=(0.5, 0.999))
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr, betas=(0.5, 0.999))
#generator_optim = torch.optim.RMSprop(generator.parameters(), lr)
#discriminator_optim = torch.optim.RMSprop(discriminator.parameters(), lr)






discriminator_loss, generator_loss = [], []

num_epochs = 50
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
    

    # Use generated data
    x_fake.data = x_fake.data.cpu()
    
    fid_score =calculate_fid(x_fake.data.numpy().reshape(batch_size,-1),x_true.data.cpu().numpy().reshape(batch_size,-1))
    print("FID score: {fid:.2f}".format(fid=fid_score))
        
    wandb.log({'epoch': epoch+1, 'Generator loss': g_loss, 'Discriminator loss': d_loss, 'FID Score': fid_score})

    wandb.watch(discriminator)
    wandb.watch(generator)