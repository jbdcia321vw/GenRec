import yaml
import os
from preprocessors.preprocessor import Preprocessor
import torch
from torch.utils.data import DataLoader
from utils import get_logger
from emb_dataset import EmbDataset
from quantizers.rqvae.trainer import Trainer
from importlib import import_module
config = yaml.safe_load(open('config.yaml','r'))
logger = get_logger(prefix='Main')


preprocessor = Preprocessor(config)
preprocessor.init()

dataset = EmbDataset(preprocessor.processed_file_path['emb'])
dataloader = DataLoader(dataset,batch_size=256,shuffle=True)

quantizer_config = yaml.safe_load(open(config['quantizer_config'],'r'))

model = getattr(import_module('quantizers.'+config['quantizer_module']),config['quantizer_class'])(quantizer_config)


trainer = Trainer(model, len(dataloader),quantizer_config)
best_loss, best_collision_rate = trainer.fit(dataloader)


logger.info(f"Best Loss: {best_loss}")
logger.info(f"Best Collision Rate: {best_collision_rate}")