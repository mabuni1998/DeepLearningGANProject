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


# The digit classes to use, these need to be in order because
# we are using one-hot representation
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


batch_size = 64
# The loaders perform the actual work
train_loader = DataLoader(dset_train, batch_size=batch_size,
                          sampler=stratified_sampler(dset_train.train_labels), pin_memory=cuda,drop_last=True)
test_loader  = DataLoader(dset_test, batch_size=batch_size, 
                          sampler=stratified_sampler(dset_test.test_labels), pin_memory=cuda)

from torch import nn

latent_dim = 100

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
    spectral_norm(nn.Conv2d(1, 64, kernel_size=4, stride=2)),
    nn.LeakyReLU(0.2),
    spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2)),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2),
    spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2)),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2),
    Flatten(),
    spectral_norm(nn.Linear(256, 1)),
    nn.Sigmoid()
).to(device)

loss = nn.BCELoss()
print("Using device:", device)

#generator_optim = torch.optim.Adam(generator.parameters(), 2e-4, betas=(0.5, 0.999))
#discriminator_optim = torch.optim.Adam(discriminator.parameters(), 2e-4, betas=(0.5, 0.999))

lr= 0.00005
generator_optim = torch.optim.RMSprop(generator.parameters(), lr)
discriminator_optim = torch.optim.RMSprop(discriminator.parameters(), lr)

from scipy.linalg import sqrtm
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

from torch.autograd import Variable
import os

tmp_img = "tmp_gan_out.png"
discriminator_loss, generator_loss = [], []

num_epochs = 50
for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss = [], []
    
    for x,_ in train_loader:
        for i in range(5):
          discriminator.zero_grad()
          #Get real data
          x_true = Variable(x).to(device)

          #Get fake data
          z = torch.randn(batch_size, latent_dim,1,1)
          z = Variable(z, requires_grad=False).to(device)
          x_fake = generator(z)

          output_true = discriminator(x_true)
          output_fake = discriminator(x_fake.detach())
          
          #If DCGAN
          #d_loss = - (1 - output_fake).log().mean() - output_true.log().mean()
          #If wasserstein:
          d_loss = -(output_true  - output_fake).mean()
          d_loss.backward()
          discriminator_optim.step()
        
          output_fake = discriminator(x_fake)
          generator.zero_grad()
        #If wasserstein:
        g_loss = -(output_fake).mean()
        #If DCGAN
        #g_loss = (1 - output_fake).log().mean()
        g_loss.backward()
        generator_optim.step()
        
        batch_d_loss.append((d_loss).item())
        batch_g_loss.append(g_loss.item())

    discriminator_loss.append(np.mean(batch_d_loss))
    generator_loss.append(np.mean(batch_g_loss))
    
    
    
    # Generate data
    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim, 1, 1)
        z = Variable(z, requires_grad=False).to(device)
        x_fake = generator(z)
    x_fake.data = x_fake.data.cpu()
    
    fid_score =calculate_fid(x_fake.data.numpy().reshape(batch_size,-1),x_true.data.cpu().numpy().reshape(batch_size,-1))
    print("FID score: {fid:.2f}".format(fid=fid_score))
    
    
    # -- Plotting --
    f, axarr = plt.subplots(1, 2, figsize=(18, 7))

    # Loss
    ax = axarr[0]
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.plot(np.arange(epoch+1), discriminator_loss)
    ax.plot(np.arange(epoch+1), generator_loss, linestyle="--")
    ax.legend(['Discriminator', 'Generator'])
    
    # Latent space samples
    ax = axarr[1]
    ax.set_title('Samples from generator')
    ax.axis('off')

    rows, columns = 8, 8
  
    canvas = np.zeros((28*rows, columns*28))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_fake.data[idx]
    ax.imshow(canvas, cmap='gray')
    
    plt.savefig(tmp_img)
    plt.close(f)
