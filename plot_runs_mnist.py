import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
cwd = os.getcwd()

font = {'family':'serif','size':20}
matplotlib.rc('font',**font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'


df_GAN = pd.read_csv(cwd+"/data_for_plots/mnist_GAN.csv")
df_spectralnorm = pd.read_csv(cwd+"/data_for_plots/mnist_spectralnorm.csv")
df_weightclipping = pd.read_csv(cwd+"/data_for_plots/mnist_weightclipping.csv")

data_list = [df_GAN,df_spectralnorm,df_weightclipping]
epochs=df_GAN["Step"].to_numpy()

mean_list = []
std_list = []
for d in data_list:
    mean_list.append(d.mean(axis=1).to_numpy())
    std_list.append(d.std(axis=1).to_numpy())
 
colors = ["red","blue","green"]
legends=["GAN","WGAN-SN","WGAN-WC"]
fig,ax = plt.subplots(1,1,figsize=(9,4.5))
plt.subplots_adjust()
for i in range(len(data_list)):    
    ax.plot(epochs,mean_list[i],color=colors[i],label=legends[i])
    ax.fill_between(epochs, mean_list[i]-std_list[i], mean_list[i]+std_list[i],color=colors[i],alpha=0.4)
ax.legend(loc="best")
ax.set_xlabel("Epochs")
ax.set_ylabel("FID-Score")
ax.set_title("MNIST")
plt.tight_layout()