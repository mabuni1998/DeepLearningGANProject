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

def get_gan_models(modelname,dataset,config):
    latent_dim = config["latent_dim"]
    nch = config["nch"]
    
    def identity_model(model):
        return model
    
    
    def layer_to_batchnorm(dims):
        return nn.BatchNorm2d(dims[0])
         
    
    def initialize_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    if modelname=="spec":
        norm = spectral_norm
        bn = nn.LayerNorm
        sm = torch.nn.Identity()
    elif modelname =="weight":
        norm = identity_model
        bn = layer_to_batchnorm
        sm = torch.nn.Identity()
    elif modelname == 'GP':
        norm = identity_model
        bn = layer_to_batchnorm
        sm = torch.nn.Identity()
    else:
        norm = identity_model
        bn = layer_to_batchnorm
        sm = nn.Sigmoid()
    
    if dataset=="mnist":
        generator = nn.Sequential(
            # nn.ConvTranspose2d can be seen as the inverse operation
            # of Conv2d, where after convolution we arrive at an
            # upscaled image.
            nn.ConvTranspose2d(latent_dim, 4*nch, kernel_size=3, stride=2,bias=False),
            nn.BatchNorm2d(4*nch),
            nn.ReLU(True),
            nn.ConvTranspose2d(4*nch, 2*nch, kernel_size=3, stride=2,bias=False),
            nn.BatchNorm2d(2*nch),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*nch, nch, kernel_size=2, stride=2,bias=False),
            nn.BatchNorm2d(nch),
            nn.ReLU(True),
            nn.ConvTranspose2d(nch, 1, kernel_size=2, stride=2),
            nn.Tanh()
        ).to(device)
        
        
        # The discriminator takes an image (real or fake)
        # and decides whether it is generated or not.
        discriminator = nn.Sequential(
            norm(nn.Conv2d(1, nch, kernel_size=2, stride=2)),
            nn.ReLU(True),
            norm(nn.Conv2d(nch, 2*nch, kernel_size=2, stride=2,bias=False)),
            bn([2*nch,7,7]),
            nn.ReLU(True),
            norm(nn.Conv2d(2*nch, 4*nch, kernel_size=3, stride=2,bias=False)),
            bn([4*nch,3,3]),
            nn.ReLU(True),
            norm(nn.Conv2d(4*nch, 1, kernel_size=3, stride=2,bias=False)),
            sm
        ).to(device)
        

    elif dataset=='CIFAR10':
        generator = nn.Sequential(
            # nn.ConvTranspose2d can be seen as the inverse operation
            # of Conv2d, where after convolution we arrive at an upscaled image.
            nn.ConvTranspose2d(latent_dim, 16*nch, kernel_size=4, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(16*nch),
            nn.ReLU(True),
            nn.ConvTranspose2d(16*nch, 8*nch, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(8*nch),
            nn.ReLU(True),
            nn.ConvTranspose2d(8*nch, 4*nch, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(4*nch),
            nn.ReLU(True),
            nn.ConvTranspose2d(4*nch, 3, kernel_size=4, stride=2,padding=1,bias=False),
            nn.Tanh()
        ).to(device)

        # The discriminator takes an image (real or fake)
        # and decides whether it is generated or not.
        discriminator = nn.Sequential(
            norm(nn.Conv2d(3, 4*nch, kernel_size=4, stride=2,padding=1,bias=False)),
            nn.ReLU(True),
            
            norm(nn.Conv2d(4*nch, 8*nch, kernel_size=4, stride=2,padding=1,bias=False)),
            bn([8*nch,8,8]),
            nn.ReLU(True),
            #nn.Dropout(0.4),
            norm(nn.Conv2d(8*nch, 8*nch, kernel_size=4, stride=2,padding=1,bias=False)),
            bn([8*nch,4,4]),
            nn.ReLU(True),
            
            norm(nn.Conv2d(8*nch,1, kernel_size=4, stride=1,padding=0,bias=True)),
            sm
        ).to(device)
    
    elif dataset=='celeba':
        generator = nn.Sequential(
            # nn.ConvTranspose2d can be seen as the inverse operation
            # of Conv2d, where after convolution we arrive at an upscaled image.
            nn.ConvTranspose2d(latent_dim, 16*nch, kernel_size=4, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(16*nch),
            nn.ReLU(True),
            nn.ConvTranspose2d(16*nch, 8*nch, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(8*nch),
            nn.ReLU(True),
            nn.ConvTranspose2d(8*nch, 4*nch, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(4*nch),
            nn.ReLU(True),
            nn.ConvTranspose2d(4*nch, 2*nch, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(2*nch),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*nch, 3, kernel_size=4, stride=2,padding=1,bias=True),
            nn.Tanh()
        ).to(device)

        # The discriminator takes an image (real or fake)
        # and decides whether it is generated or not.
        discriminator = nn.Sequential(
            norm(nn.Conv2d(3, 2*nch, kernel_size=4, stride=2,padding=1,bias=False)),
            nn.ReLU(True),
            
            norm(nn.Conv2d(2*nch, 4*nch, kernel_size=4, stride=2,padding=1,bias=False)),
            bn([4*nch,16,16]),
            nn.ReLU(True),
            #nn.Dropout(0.4),
            norm(nn.Conv2d(4*nch, 8*nch, kernel_size=4, stride=2,padding=1,bias=False)),
            bn([8*nch,8,8]),
            nn.ReLU(True),
            
            norm(nn.Conv2d(8*nch,8*nch, kernel_size=4, stride=2,padding=1,bias=False)),
            bn([8*nch,4,4]),
            nn.ReLU(True),
            norm(nn.Conv2d(8*nch,1, kernel_size=4, stride=1,padding=0,bias=True)),
            sm
        ).to(device)


    return generator,discriminator
