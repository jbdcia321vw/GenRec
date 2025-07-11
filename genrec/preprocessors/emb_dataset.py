from torch.utils.data import Dataset
import numpy as np

class EmbDataset(Dataset):
    def __init__(self,file_path):
        self.embeddings = np.load(file_path)
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self,index):
        return self.embeddings[index]