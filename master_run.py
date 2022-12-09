import sys
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST,CIFAR10,CelebA
from torchvision.transforms import ToTensor, Normalize, Compose,Resize
from functools import reduce
from torch import nn
import os
from FID_score import calculate_fid
from models_file import get_gan_models

wandb.config = {
  "learning_rate": 2e-4,
  "epochs": 50,
  "batch_size": 64,
  "latent_dim":100,
  "clipping":0.02,
  "nch":12,
  "lambda_gp":10
}
config = wandb.config


dataset, modelname = sys.argv[1],sys.argv[2]

#Define model
latent_dim = config["latent_dim"]
nch = config["nch"]
batch_size = config["batch_size"]


generator, discriminator = get_gan_models(modelname,dataset,config)

def do_nothing(discriminator):
    "Nothing"

def clip_grad(discriminator):
    with torch.no_grad():
        for param in discriminator.parameters():
            param.clamp_(-config["clipping"], config["clipping"])

def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def wasserstein_loss(labels, output):
    return torch.mean(labels * output)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.autograd.Variable(Tensor(real_samples.shape[0], 1,1,1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

if modelname == 'weight':
    clipping = clip_grad
    project_name="WGAN_weightclipping"
    n_train=5
    loss_fn = wasserstein_loss
    real_labels = -torch.ones(batch_size, 1, 1, 1).to(device)
    fake_labels = torch.ones(batch_size, 1, 1, 1).to(device)

elif modelname =='spec':
    project_name="WGAN_spectralnorm"
    clipping = do_nothing
    n_train=5
    loss_fn = wasserstein_loss
    real_labels = -torch.ones(batch_size, 1, 1, 1).to(device)
    fake_labels = torch.ones(batch_size, 1, 1, 1).to(device)

elif modelname=='GP':
    project_name="WGAN_GP"
    clipping = do_nothing
    n_train=5
    loss_fn = wasserstein_loss
    real_labels = -torch.ones(batch_size, 1, 1, 1).to(device)
    fake_labels = torch.ones(batch_size, 1, 1, 1).to(device)
else:
    project_name="GAN"
    clipping = do_nothing
    n_train = 1
    loss_fn = nn.BCELoss()
    real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)


if dataset == 'mnist':
    project_dataset_name = 'mnist-'
    # The digit classes to use, these need to be in order because
    classes = np.arange(10)

    def one_hot(labels):
        y = torch.eye(len(classes)) 
        return y[labels]

    dset_train = MNIST("./", train=True, download=True, transform=Compose([ToTensor(),Normalize(0.5,0.5)]), target_transform=one_hot)
    
    def stratified_sampler(labels):
        """Sampler that only picks datapoints corresponding to the specified classes"""
        (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
        indices = torch.from_numpy(indices)
        return SubsetRandomSampler(indices)

    train_loader = DataLoader(dset_train, batch_size=batch_size,
                              sampler=stratified_sampler(dset_train.targets), pin_memory=cuda,drop_last=True,num_workers=1)
    
elif dataset=='CIFAR10':
    project_dataset_name = 'CIFAR10_'
    
    transform = Compose(
        [ToTensor(),
        Normalize((0.5, 0.5 ,0.5), (0.5,0.5,0.5)),  # subtract 0.5 and divide by 0.5
        ]
    )

    # Load datasets
    train_set = CIFAR10('./', train=True, download=False, transform=transform)
    # The loaders perform the actual work
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=cuda,drop_last=True,num_workers=1)    
    
elif dataset=='celeba':
    project_dataset_name = 'celeb_'
    transform = Compose(
        [Resize([64,64]),
         ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # subtract 0.5 and divide by 0.5
        ]
    )

    # Load datasets
    train_set = CelebA('./', download=True, transform=transform)
    # The loaders perform the actual work
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=cuda,drop_last=True,num_workers=1)    
    



run = wandb.init(project=project_dataset_name+project_name, entity="gan_project_cm",config=config)

print("Using device:", device)

lr = config["learning_rate"]
generator_optim = torch.optim.Adam(generator.parameters(), lr, betas=(0.5, 0.999))
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr, betas=(0.5, 0.999))

discriminator.apply(initialize_weights)
generator.apply(initialize_weights)

discriminator_loss, generator_loss = [], []


num_epochs = config["epochs"]
for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss = [], []
    
    for x,_ in train_loader:
        #Get real data
        x_true = x.to(device)       
        for i in range(n_train):
          #discriminator.zero_grad()
          discriminator_optim.zero_grad()
          #Get fake data
          z = torch.randn(batch_size, latent_dim,1,1)
          z.requires_grad=False
          z = z.to(device)
          x_fake = generator(z)
    
          
          output_true = discriminator(x_true)
          output_fake = discriminator(x_fake.detach())

          critic_loss_true = loss_fn(output_true, real_labels)
          critic_loss_fake = loss_fn(output_fake, fake_labels)
          
          gp = compute_gradient_penalty(discriminator, x_true.data, x_fake.data) if modelname=='GP' else 0
          d_loss = critic_loss_true  + critic_loss_fake + gp*config["lambda_gp"]
          d_loss.backward()
          #critic_loss_fake.backward()
          #critic_loss_true.backward()
          
          clipping(discriminator)
          discriminator_optim.step()
        
    
        #z = torch.randn(batch_size, latent_dim,1,1).to(device)    
        #x_fake = generator(z)
        generator_optim.zero_grad()
        output_fake = discriminator(x_fake)
        generator.zero_grad()
        g_loss = loss_fn(output_fake, real_labels)
        g_loss.backward()
        clipping(discriminator)
        generator_optim.step()
        
        
        batch_d_loss.append((d_loss).item())
        batch_g_loss.append(g_loss.item())

    discriminator_loss.append(np.mean(batch_d_loss))
    generator_loss.append(np.mean(batch_g_loss))
    

    # Use generated data
    x_fake.data = x_fake.data
    input_fake_data = torch.cat([x_fake.data, x_fake.data, x_fake.data], dim=1).to(device) if dataset=='mnist' else x_fake.data
    input_true_data = torch.cat([x_true.data, x_true.data, x_true.data], dim=1).to(device) if dataset=='mnist' else x_true.data
    
    
    fid_score =calculate_fid(input_fake_data,input_true_data)
    print("FID score: {fid:.2f}".format(fid=fid_score))
        
    wandb.log({'epoch': epoch+1, 'Generator loss': generator_loss[-1], 'Discriminator loss': discriminator_loss[-1], 'FID Score': fid_score})

#Save model
generatorpath = os.getcwd()+'/models/'+project_dataset_name+project_name+'_g.pth'
discriminatorpath = os.getcwd()+'/models/'+project_dataset_name+project_name+'_d.pth'

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

