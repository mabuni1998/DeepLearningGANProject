import matplotlib.pyplot as plt
plt.style.use(["seaborn-deep", "seaborn-whitegrid"])
import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
#device = 'cpu'
import os
import seaborn as sns
sns.set_style("whitegrid")
from torchvision.utils import make_grid
from models_file import get_gan_models
import matplotlib
matplotlib.use('Agg')
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

#Load celeba
generator_celeb,d = get_gan_models('GP','celeba',config)
generatorpath = os.getcwd()+'/models/celeba_WGAN_GP_g.pth'
generator_celeb.load_state_dict(torch.load(generatorpath,map_location=device))



#Load cifar CIFAR10_g.pth'
generator_cifar,d = get_gan_models('GP','CIFAR10',config)
generatorpath = os.getcwd()+'/models/CIFAR10_WGAN_GP_g.pth'
generator_cifar.load_state_dict(torch.load(generatorpath,map_location=device))


#Load mnist generator
generator_mnist,d = get_gan_models('gan','mnist',config)
generatorpath = os.getcwd()+'/models/mnist_GAN_g.pth'
generator_mnist.load_state_dict(torch.load(generatorpath,map_location=device))

generators=[generator_mnist,generator_cifar,generator_celeb]

def show_image(ax,img,title):
    img = img.detach().cpu()
    img = img / 2 + 0.5   # unnormalize
    with sns.axes_style("white"):
        ax.imshow(img.permute((1, 2, 0)).numpy())
        ax.axis('off')
        ax.set_title(title)
titles=['MNIST - GAN','CIFAR10 - WGAN GP','CelebA - WGAN GP']
fig,ax = plt.subplots(3,1,figsize=(9,23))
for i,generator in enumerate(generators):
    # Generate data
    with torch.no_grad():
        z = torch.randn(64, 100, 1, 1)
        z = z.to(device)
        z.requires_grad=False
        x_fake = generator(z)
    x_fake.data = x_fake.data.cpu()
    show_image(ax[i],make_grid(x_fake,nrow=8),titles[i])
plt.tight_layout()
plt.savefig(os.getcwd()+'/plots/samples.pdf')

