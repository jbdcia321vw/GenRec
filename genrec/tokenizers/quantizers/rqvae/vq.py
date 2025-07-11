import torch
from .levels import kmeans
import torch.nn as nn
import torch.nn.functional as F
class VectorQuantizer(nn.Module):
    def __init__(self,codebook_size,e_dim,beta=0.5,kmeans_init=False,kmeans_iters=10,metric="euclidean"):
        super().__init__()
        self.beta = beta
        self.embedding = nn.Embedding(codebook_size,e_dim)
        self.metric = metric
        self.inited = False
        self.codebook_size=codebook_size
        self.kmeans_iters=kmeans_iters
        if not kmeans_init:
            nn.init.uniform_(self.embedding.weight,-1.0 / self.codebook_size, 1.0 / self.codebook_size)
            self.inited=True
    def get_codebook(self):
        return self.embedding.weight
    def get_codebook_entry(self,indices):
        return self.embedding(indices)
    def distance(self,x):
        eps = torch.tensor(1e-8).to(x.device)
        if self.metric=="euclidean":
            dis = torch.sum(x**2,dim=1,keepdim=True)+ \
            torch.sum(self.embedding.weight**2,dim=1,keepdim=True).t()- \
            2*torch.matmul(x,self.embedding.weight.t())
            return torch.sqrt(dis.clamp(min=eps))
        elif self.metric=="cosine":
            x_norm = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True)) + eps
            e_norm = torch.sqrt(torch.sum(self.embedding.weight**2,dim=1,keepdim=True)).t() + eps
            sim = torch.matmul(x,self.embedding.weight.t())/(x_norm*e_norm)
            sim = torch.clamp(sim,-1,1)
            dis = 1-sim
            return dis
    def forward(self,x):
        '''
        @param x: (B,e_dim)
        @return:
            quantized: (B,e_dim)
            indices: (B)
            vq_loss: (B)
        '''
        #x.shape=(B,e_dim)
        #return (B,e_dim),(B),(B)
        if not self.inited and self.training:
            self.embedding.weight.data.copy_(kmeans(x,self.codebook_size,self.kmeans_iters))
            self.inited=True
        dist = self.distance(x)
        indices = dist.argmin(dim=-1)
        quantized = self.get_codebook_entry(indices)

        vq_loss = self.vq_loss(x,quantized)

        quantized = x + (quantized - x).detach()

        return quantized,indices,vq_loss
    def vq_loss(self,x,quantized):
        #x.shape=(B,e_dim)
        #quantized.shape=(B,e_dim)
        #return (B)
        commitment_loss = F.mse_loss(quantized.detach(),x)
        codebook_loss = F.mse_loss(quantized,x.detach())
        return commitment_loss+self.beta*codebook_loss
