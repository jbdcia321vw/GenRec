import os
import logging
def yield_json(file_path):
        with open(file_path,'r') as f:
            for line in f.readlines():
                yield(eval(line.strip()))
def get_logger(log_path=None):
     logger = logging.getLogger(__name__)
     logger.setLevel(logging.INFO)
     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
     console_handler = logging.StreamHandler()
     console_handler.setLevel(logging.INFO)
     console_handler.setFormatter(formatter)
     logger.addHandler(console_handler)
     return logger