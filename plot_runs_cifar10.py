import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
cwd = os.getcwd()

font = {'family':'serif','size':20}
matplotlib.rc('font',**font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'


def load_data(dataset):
    df_GAN = pd.read_csv(cwd+"/data_for_plots/{s:s}_GAN.csv".format(s=dataset))
    df_spectralnorm = pd.read_csv(cwd+"/data_for_plots/{s:s}_spec.csv".format(s=dataset))
    df_weightclipping = pd.read_csv(cwd+"/data_for_plots/{s:s}_weight.csv".format(s=dataset))
    df_GP = pd.read_csv(cwd+"/data_for_plots/{s:s}_GP.csv".format(s=dataset))
    
    
    data_list = [df_GAN,df_spectralnorm,df_weightclipping,df_GP]
    epochs=df_GAN["Step"].to_numpy()
    
    mean_list = []
    std_list = []
    for d in data_list:
        mean_list.append(d.mean(axis=1).to_numpy())
        std_list.append(d.std(axis=1).to_numpy())
    return mean_list,std_list,epochs
        
 
datasets = ['MNIST', 'CIFAR10', 'CelebA']
colors = ["red","blue","green",'darkorange']
legends=["GAN","WGAN-SN","WGAN-WC","WGAN-GP"]
fig,ax = plt.subplots(1,3,figsize=(18,5.5))
plt.subplots_adjust()
for j in range(len(datasets)): 
    mean_list, std_list,epochs = load_data(datasets[j])
    for i in range(len(mean_list)):    
        ax[j].plot(epochs,mean_list[i],color=colors[i],label=legends[i])
        ax[j].fill_between(epochs, mean_list[i]-std_list[i], mean_list[i]+std_list[i],color=colors[i],alpha=0.4)
    ax[j].legend(loc="best")
    ax[j].set_xlabel("Epochs")
    ax[j].set_title(datasets[j]+" - 10 experiments")
ax[0].set_ylim(0,80)
ax[0].set_ylabel("FID-Score")
plt.tight_layout()
plt.savefig(cwd+"/plots/runs.svg")