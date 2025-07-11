import torch
import torch.optim as optim
from time import time
import os
from preprocessors.utils import set_color,delete_file,get_logger
import heapq
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import numpy as np
class Trainer():
    def __init__(self,model,data_num,config):
        self.config = config
        self.model=model
        self.device=torch.device(config['device'])
        self.optimizer=config['optimizer']
        self.lr=config['lr']
        self.weight_decay=config['weight_decay']
        self.lr_scheduler_type = config['lr_scheduler_type']
        self.epochs = config['epochs']
        self.warmup_steps = config['warmup_epochs']*data_num
        self.max_steps=config['epochs']*data_num
        self.eval_step=min(config['eval_step'],self.epochs)
        self.ckpt_dir=config['ckpt_dir']
        saved_mode_dir=f'{int(time())}'
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_mode_dir)
        os.makedirs(self.ckpt_dir,exist_ok=True)

        self.save_limit = config['save_limit']
        self.best_save_heap = []
        self.newest_save_queue = []

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt="best_loss_model.pth"
        self.best_collision_ckpt="best_collision_model.pth"

        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

        self.logger = get_logger(prefix='[Trainer]')
        #self.save_limit = config['save_limit']
    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.optimizer
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _train_epoch(self,train_data,epoch_idx):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        iter_data = tqdm(train_data,total=len(train_data),ncols=100)
        for batch,data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                y,rq_loss = self.model(data)
                loss,recon_loss=self.model.compute_loss(data,y,rq_loss)
                if torch.isnan(loss):
                    raise ValueError('loss is nan')
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
        return total_loss,total_recon_loss
    @torch.no_grad()
    def _valid_epoch(self,valid_data):
        self.model.eval()
        iter_data = tqdm(valid_data,total=len(valid_data),ncols=100)
        indices_set = set()
        num_sample = 0
        for batch,data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)
        collision_rate = (num_sample-len(list(indices_set)))/num_sample
        return collision_rate
    def _save_checkpoint(self,epoch,collision_rate=1,ckpt_file=None):
        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir,f"epoch_{epoch}_collision_{collision_rate:.4f}.pth")
        state = {
            "config":self.config,
            "epoch": epoch,
            "best_loss":self.best_loss,
            "best_collision_rate":self.best_collision_rate,
            "state_dict":self.model.state_dict(),
            "optimizer":self.optimizer.state_dict()
        }        
        torch.save(state,ckpt_path,pickle_protocol=4)

        self.logger.info(f"save checkpoint to {ckpt_path}")
        return ckpt_path
    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"
    def fit(self,data):
        cur_eval_step=0
        
        for epoch_idx in range(self.epochs):
            training_start_time = time()
            train_loss,train_recon_loss = self._train_epoch(data,epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx,training_start_time,training_end_time,train_loss,train_recon_loss
            )
            self.logger.info(train_loss_output)

            if (epoch_idx + 1) % self.eval_step==0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if train_loss<self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch_idx,collision_rate,self.best_loss_ckpt)
                if collision_rate<self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    #cur_eval_step=0
                    self._save_checkpoint(epoch_idx,collision_rate,self.best_collision_ckpt)

                valid_end_time = time()

                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])