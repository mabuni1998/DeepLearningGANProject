from torchvision import transforms,models
import torch
from scipy.linalg import sqrtm
import numpy as np
from os import getcwd
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
cwd = getcwd()


def calculate_fid(train, target):
    
    #Functions for calculating FID

    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    inception_mdl = models.inception_v3(init_weights=True)

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        inception_mdl = inception_mdl.cuda()

    inception_mdl.load_state_dict(torch.load(cwd+"/models/inception_v3_google-0cc3c7bd.pth"))  
    inception_mdl.eval();
    # extract train and eval layers from the model
    train_nodes, eval_nodes = get_graph_node_names(inception_mdl)

    # remove the last layer
    return_nodes = eval_nodes[:-1]

    # create a feature extractor for each intermediary layer
    feat_inception = create_feature_extractor(inception_mdl, return_nodes=return_nodes)
    if is_cuda:
        feat_inception = feat_inception.cuda()
        
        
    train = preprocess(train)
    target = preprocess(target)
    train = feat_inception(train)
    target = feat_inception(target)
    target = target['flatten'].cpu().detach().numpy()
    train = train['flatten'].cpu().detach().numpy()
    
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
