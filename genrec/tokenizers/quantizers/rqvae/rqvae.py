import torch
import torch.nn as nn
import torch.nn.functional as F
from .rq import ResidualQuantizer
from sklearn.preprocessing import StandardScaler
from .levels import MLPLayers
from sklearn.decomposition import PCA
import numpy as np
class RQVAE(nn.Module):
    def __init__(self,config):
        if config['kmeans_init']:
            assert config['kmeans_iters'] is not None
        super().__init__()
        self.loss_type = config['loss_type']
        self.mu = config['mu']
        layers = [config['in_dim']] + config['layers'] + [config['e_dim']]
        
        self.encoder = MLPLayers(layers,config['dropout_prob'],bn=config['bn'])
        self.decoder = MLPLayers(layers[::-1],config['dropout_prob'],bn=config['bn'])
        self.rq = ResidualQuantizer(config['codebook_size'],config['e_dim'],config['beta'],config['kmeans_init'],config['kmeans_iters'],config['metric_list'])
    def forward(self,x):
        '''
        @param x: (B,in_dim)
        @return:
        y: (B,in_dim)
        mean_loss: (1)
        '''
        #x = self.scaler.transform(x.cpu().detach().numpy())
        #x = self.encoder(torch.from_numpy(x).to(torch.device('cuda')))
        x = self.encoder(x)
        y,indices,rq_loss = self.rq(x)
        y = self.decoder(y)
        return y,rq_loss
    @torch.no_grad()
    def get_indices(self,x):
        x = self.encoder(x)
        _,indices,_ = self.rq(x)
        return indices
    def compute_loss(self,x,y,rq_loss):
        if self.loss_type=='mse':
            recon = F.mse_loss(y,x,reduction='mean')
        elif self.loss_type =='l1':
            recon = F.l1_loss(y,x,reduction='mean')
        else:
            raise NotImplementedError
        total_loss = recon + self.mu * rq_loss
        return total_loss,recon