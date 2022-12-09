import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import pandas as pd
cwd = os.getcwd()

font = {'family':'serif','size':20}
matplotlib.rc('font',**font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'


def load_mean(dataset):
    df_GAN = pd.read_csv(cwd+"/data_for_plots/{s:s}_GAN.csv".format(s=dataset))
    df_spectralnorm = pd.read_csv(cwd+"/data_for_plots/{s:s}_spec.csv".format(s=dataset))
    df_weightclipping = pd.read_csv(cwd+"/data_for_plots/{s:s}_weight.csv".format(s=dataset))
    df_GP = pd.read_csv(cwd+"/data_for_plots/{s:s}_GP.csv".format(s=dataset))

    data_list = [df_GAN,df_spectralnorm,df_weightclipping,df_GP]
    
    mean_list = []

    for d in data_list:
        mean_list.append(d.mean(axis=1).to_numpy()[-1])
    return mean_list  

def load_model_means():
    mnist_list = load_mean('MNIST')
    cifar10_list = load_mean('CIFAR10')
    celeba_list = load_mean('CelebA')
    out = []
    for i in range(len(mnist_list)):    
        out.append([mnist_list[i],cifar10_list[i],celeba_list[i]])
    return out[0],out[1],out[2],out[3]
datasets = ['MNIST', 'CIFAR10', 'CelebA']
gan,wgan_spec,wgan_wc, wgan_gp = load_model_means()

X = np.arange(3)
colors = ["red","blue","green",'darkorange']
fig,ax = plt.subplots(1,1,figsize=(9,5.5))
ax.bar(X+0.0,gan,width=0.2,color=colors[0],label='GAN')
ax.bar(X+0.2,wgan_spec,width=0.2,color=colors[1],label='WGAN-SN')
ax.bar(X+0.4,wgan_wc,width=0.2,color=colors[2],label='WGAN-WC')
ax.bar(X+0.6,wgan_gp,width=0.2,color=colors[3],label='WGAN-GP')
ax.set_xticks(X+0.3, datasets)
ax.set_ylabel('FID Score')
ax.legend(numpoints=1,ncol=5,loc='lower left', bbox_to_anchor=[-0.057,1.01],columnspacing=0.7,handlelength=1,handletextpad=0.4)
plt.tight_layout()
plt.savefig(cwd + "/plots/histogram.svg")