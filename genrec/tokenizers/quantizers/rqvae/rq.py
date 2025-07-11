import torch
import torch.nn as nn
import numpy as np
from .vq import VectorQuantizer
class ResidualQuantizer(nn.Module):
    def __init__(self,codebook_size,e_dim,beta,kmeans_init = False,kmeans_iters=100,metric_list=None):
        super().__init__()
        self.vq_layers = []
        n_e_list = codebook_size
        for i in range(len(n_e_list)):
            vq = VectorQuantizer(codebook_size[i],e_dim,beta,kmeans_init,kmeans_iters,metric_list[i])
            self.vq_layers.append(vq)
        self.vq_layers = nn.ModuleList(self.vq_layers)
    def get_codebook(self):
        all_codebook=[]
        for vq in self.vq_layers:
            all_codebook.append(vq.get_codebook())
        return torch.stack(all_codebook)
    def forward(self,x):
        '''
        @param x: (B,e_dim)
        @return:
        y: (B,e_dim)
        all_indices: (B,L)
        mean_loss: (1)
        '''
        residual = x
        all_indices = []
        all_loss = []
        y = 0
        for vq in self.vq_layers:
            x_q,indices,vq_loss = vq(residual)
            residual = residual - x_q
            y = y + x_q
            all_indices.append(indices)
            all_loss.append(vq_loss)
        mean_loss = torch.stack(all_loss).mean()
        all_indices = torch.stack(all_indices,dim=-1)
        return y,all_indices,mean_loss