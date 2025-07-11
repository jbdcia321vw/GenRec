from torch.utils.data import Dataset
import json
class GenRecDataset(Dataset):
    def __init__(self,config,mode='train'):
        self.config = config
        self.data = []
        self.mode = mode
        self._load_data()
    def _load_data(self):
        inters = json.load(open(self.config.processed_file_path['inter'],'r'))
        indices = json.load(open(self.config.processed_file_path['indices'],'r'))
        if self.mode=='train':
            self._load_train_data(inters,indices)
    def _load_train_data(self,inters,indices):
        for user,items in inters.items():
            for i in range(1,len(items)-2):
                one_data = {}
                one_data['item']=indices[items[i]]
                one_data['inter']=sum(indices[items[:i]],[])
                if self.config.history > 0:
                    one_data['inter']=one_data['inter'][-self.config.history:]
                self.data.append(one_data)
    def _load_valid_data(self,inters,indices):
        for user,items in inters.items():
            one_data = {}
            one_data['item']=indices[items[-2]]
            one_data['inter']=sum(indices[items[:-2]],[])
            if self.config.history > 0:
                one_data['inter']=one_data['inter'][-self.config.history:]
            self.data.append(one_data)
    def _load_test_data(self,inters,indices):
        for user,items in inters.items():
            one_data = {}
            one_data['item']=indices[items[-1]]
            one_data['inter']=sum(indices[items[:-1]],[])
            if self.config.history > 0:
                one_data['inter']=one_data['inter'][-self.config.history:]
            self.data.append(one_data)
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index]