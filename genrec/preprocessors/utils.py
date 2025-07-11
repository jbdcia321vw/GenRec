import os
import logging
import requests
import gzip
import yaml
from time import time
def yield_json(file_path):
        with open(file_path,'r') as f:
            for line in f.readlines():
                yield(eval(line.strip()))
def yield_gzip(file_path):
    with gzip.open(file_path,'r') as f:
        for line in f.readlines():
            yield(eval(line.strip()))
def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"
def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
def get_logger(prefix=''):
     if hasattr(get_logger, '_logger'):
        formatter = logging.Formatter(f'{prefix} %(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for handler in get_logger._logger.handlers:
            handler.setFormatter(formatter)
        return get_logger._logger
     log_path = yaml.safe_load(open('./config.yaml','r'))['log_path']
     logger = logging.getLogger('Main')
     logger.setLevel(logging.INFO)
     formatter = logging.Formatter(f'{prefix} %(asctime)s - %(name)s - %(levelname)s - %(message)s')
     if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
     console_handler = logging.StreamHandler()
     console_handler.setLevel(logging.INFO)
     console_handler.setFormatter(formatter)
     logger.addHandler(console_handler)
     get_logger._logger = logger
     return logger

def download_file(url,save_path):
    res = requests.get(url)
    with open(save_path,'wb') as f:
        f.write(res.content)
    