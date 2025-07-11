import numpy as np
from sklearn.cluster import KMeans

class ProductQuantization():
    def __init__(self,n_digits:int,n_clusters:int):
        self.n_clusters=n_clusters
        self.n_digits=n_digits
    def fit(self,x:np.array):
        assert x.shape[-1]%self.n_digits==0,"vector dim can't be divided by n_digits"
        assert x.shape[0]>=self.n_clusters,"number of vector can't less than n_cluster"
        digits = self._get_digits(x)
        self.quantizers = []
        for i in range(self.n_digits):
            self.quantizers.append(KMeans(n_clusters=self.n_clusters).fit(digits[:,i,:]))
    def encode(self,x:np.array):
        assert hasattr(self,'quantizers') and self.quantizers, "Must call fit() before encode()"
        assert x.shape[-1]%self.n_digits==0,"vector dim can't be divided by n_digits"
        digits = self._get_digits(x)
        output = np.zeros((x.shape[0],self.n_digits),dtype=np.int32)
        for i in range(self.n_digits):
            output[:,i]=self.quantizers[i].predict(digits[:,i,:])
        return output
    def _get_digits(self,x):
        dim = x.shape[-1]//self.n_digits
        digits = np.zeros((x.shape[0],self.n_digits,dim))
        for i in range(self.n_digits):
            digits[:,i,:]=x[:,i*dim:(i+1)*dim]
        return digits