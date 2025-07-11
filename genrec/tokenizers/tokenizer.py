import os
from collections import defaultdict
import json
from utils import yield_json
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from importlib import import_module
import yaml
class Tokenizer():
    def __init__(self,config):
        self.config=config
        self.device = torch.device(config['device'])
        self.cache = config['cache']
        self.dataset_name = config['dataset_name']
        self.raw_file_path = {
            'meta':os.path.join(self.cache,'raw',f'meta_{self.dataset_name}.json'),
            'inter':os.path.join(self.cache,'raw',f'{self.dataset_name}_5.json')
        }
        self.processed_file_path = {
            'meta':os.path.join(self.cache,'processed',f'{self.dataset_name}/meta.json'),
            'inter':os.path.join(self.cache,'processed',f'{self.dataset_name}/inter.json'),
            'emb':os.path.join(self.cache,'processed',f'{self.dataset_name}/emb.npy'),
            'maps':os.path.join(self.cache,'processed',f'{self.dataset_name}/maps.json'),
            'indices':os.path.join(self.cache,'processed',f'{self.dataset_name}/indices.npy')
        }
        self.feat_cols = config['feat_cols']
        self.model_name = config['embedding_model_name']
        self.maps = {
            'asin2id':dict(),
            'id2asin':list()
        }


    
    def _convert_feat(self,feat):
        if isinstance(feat,str):
            return feat.strip()
        elif isinstance(feat,list):
            return ','.join(feat[0]).strip()#convert categories to string
        else:
            return str(feat.strip())
    def _process_meta(self,file_path):
        meta = defaultdict(dict)
        for item in yield_json(file_path):
            text = {}
            for col in item:
                if col in self.feat_cols:
                    text[col] = self._convert_feat(item[col])
            meta[item['asin']]=text
        with open(self.processed_file_path['meta'],'w') as f:
            json.dump(meta,f,indent=4)
    def _process_inter(self,file_path):
        inters = defaultdict(list)
        for inter in yield_json(file_path):
            reviewerID = inter['reviewerID']
            asin = inter['asin']
            reviewTime = inter['reviewTime']
            year = int(reviewTime[-4:])
            month = int(reviewTime[:2])
            day = int(reviewTime.split(',')[0][-2:].strip())
            reviewTime = year*10000+month*100+day
            inters[reviewerID].append((asin,reviewTime))
        for reviewerID in inters:
            inters[reviewerID]=[self.maps['asin2id'][asin] for asin,_ in sorted(inters[reviewerID],key=lambda x:x[1])]
        with open(self.processed_file_path['inter'],'w') as f:
            json.dump(inters,f,indent=4)
    def _process_raw(self):
        if os.path.exists(self.processed_file_path['meta']) and os.path.exists(self.processed_file_path['inter']):
            print('raw file exists, skip processing')
            return
        self._process_meta(self.raw_file_path['meta'])
        self._process_inter(self.raw_file_path['inter'])
    def _build_map(self):
        if os.path.exists(self.processed_file_path['maps']):
            print('map file exists, load from ',self.processed_file_path['maps'])
            self.maps = json.load(open(self.processed_file_path['maps'],'r'))
            return
        for item in yield_json(self.raw_file_path['meta']):
            asin = item['asin']
            self.maps['asin2id'][asin] = len(self.maps['id2asin'])
            self.maps['id2asin'].append(asin)
        with open(self.processed_file_path['maps'],'w') as f:
            json.dump(self.maps,f,indent=4)
    def _build_emb(self):
        if os.path.exists(self.processed_file_path['emb']):
            print('embedding file exists, skip building')
            return
        item_meta = []
        with open(self.processed_file_path['meta']) as f:
            meta = json.load(f)
            for text in meta.values():
                item_meta.append(','.join(text.values()))
        model = SentenceTransformer(os.path.join(self.cache,'models',self.model_name),device=torch.device(self.config['device']))
        embeddings = model.encode(item_meta,batch_size=self.config['batch_size'],show_progress_bar=self.config['show_progress_bar'],convert_to_numpy=True,device=torch.device(self.config['device']))
        np.save(self.processed_file_path['emb'],embeddings)
    def _quantize(self):
        if os.path.exists(self.processed_file_path['indices']):
            print('indices file exists, skip quantizing')
            return

        embeddings = np.load(self.processed_file_path['emb'])
        config = yaml.safe_load(open(self.config['quantizer_config'],'r'))
        quantizer = getattr(import_module('quantizers.'+self.config['quantizer_module']),self.config['quantizer_class'])(**config)
        quantizer.load_state_dict(torch.load(self.config['quantizer_ckpt_path']))
        quantizer.eval()
        quantizer.to(self.device)
        with torch.no_grad():
            indices = quantizer.get_indices(torch.from_numpy(embeddings).to(self.device))
        tokens = ['<a_{}>','<b_{}>','<c_{}>','<d_{}>','<e_{}>','<f_{}>','<g_{}>','<h_{}>','<i_{}>','<j_{}>','<k_{}>','<l_{}>','<m_{}>','<n_{}>','<o_{}>','<p_{}>','<q_{}>','<r_{}>','<s_{}>','<t_{}>','<u_{}>','<v_{}>','<w_{}>','<x_{}>','<y_{}>','<z_{}>']
        all_indices = []
        for indice in  indices:
            token = []
            for idx,i in enumerate(indice):
                token.append(tokens[idx].format(i))
            all_indices.append(token)

        json.dump(all_indices,open(self.processed_file_path['indices'],'w'),indent=4)
    def init(self):
        os.makedirs(os.path.join(self.cache,'processed',self.dataset_name),exist_ok=True)
        self._build_map()
        self._process_raw()
        self._build_emb()
        #self._quantize()