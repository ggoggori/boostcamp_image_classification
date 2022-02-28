import torch
from models.model import Network
from utils.metrics import cal_metric
from utils.utils import get_device
import numpy as np
import random
from tqdm import tqdm
import os
import time
import shutil
import sys
import math
import wandb

## to-do
'''
criterion, optimizer 두개 config로 관리하기
'''

class Trainer(object):
    def __init__(self, config:dict, dataset):
        self.config = config
        self.dataset = dataset
        self.device = get_device()
        now = time.localtime()
        self.date = f'{now.tm_mday}d-{now.tm_hour}h-{now.tm_min}m'
        self.model_checkpoint_dir = os.path.join(self.config['dir']['checkpoint_dir'], self.date)

    def train_and_validate_one_epoch(self, model, epoch, dataloaders, criterion, optimizer, best_score):
        for phase in ['train', 'valid']:
            dataloader = dataloaders[phase]
            running_loss, correct = 0,0
            targets, predictions = [], []
            total_size = len(dataloader.dataset)

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch in dataloader:
                image = batch['image'].to(self.device)
                target = batch['labels']['label'].to(self.device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    output = model(image)
                    loss = criterion(output, target)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(output, 1)

                targets.append(target)
                predictions.append(preds)

                running_loss += loss.item() * image.size(0)
                correct += torch.sum(preds==target)

            total_loss = running_loss / len(dataloader.dataset)
            targets = torch.cat(targets)
            predictions = torch.cat(predictions)
            f1score, total_acc = cal_metric(predictions, targets, total_size)
            print(f'{phase}-[EPOCH:{epoch}] |F1: {f1score:.3f} | ACC: {total_acc:.3f} | Loss: {total_loss:.5f}|')

            if phase == 'valid' and f1score > best_score:
                best_score = f1score
                print(f'{best_score:.3f} model saved')
                self._checkpoint(model, epoch, best_score)

            wandb.log({f"f1score_{phase}":f1score, f"loss_{phase}":total_loss })
        return best_score

    def train_and_validate_one_epoch_multiple(self, model, epoch, dataloaders, criterion_ce, criterion_bce, optimizer, best_score):
        for phase in ['train', 'valid']:
            dataloader = dataloaders[phase]
            running_loss, correct = 0,0
            targets, predictions = [], []
            total_size = len(dataloader.dataset)

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch in dataloader:
                image = batch['image'].to(self.device)
                gender = batch['labels']['gender'].to(self.device)
                age = batch['labels']['age'].to(self.device)
                mask = batch['labels']['mask'].to(self.device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    gender_out, age_out, mask_out  = model(image)
                    loss = criterion_bce(gender_out.squeeze(), gender.float()) + criterion_ce(age_out, age) + criterion_ce(mask_out, mask) 
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(age_out, 1)

                targets.append(age)
                predictions.append(preds)

                running_loss += loss.item() * image.size(0)

            total_loss = running_loss / len(dataloader.dataset)
            targets = torch.cat(targets)
            predictions = torch.cat(predictions)
            f1score, total_acc = cal_metric(predictions, targets, total_size)
            print(f'{phase}-[EPOCH:{epoch}] |F1: {f1score:.3f} | ACC: {total_acc:.3f} | Loss: {total_loss:.5f}|')

            if phase == 'valid' and f1score > best_score:
                best_score = f1score
                print(f'{best_score:.3f} model saved')
                self._checkpoint(model, epoch, best_score)

            wandb.log({f"f1score_{phase}":f1score, f"loss_{phase}":total_loss })

        return best_score
        
    def train(self):
        wandb.init(project='image_classification', reinit=True, config=self.config)
        wandb.run.name = self.date
        wandb.run.save()

        set_randomseed(self.config['random_seed'])
        model = Network(self.config).to(self.device)
        criterion_ce = torch.nn.CrossEntropyLoss()
        criterion_bce = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config['LR']))
        
        train_dataloader, valid_dataloader = self.dataset.make_dataloader()
        dataloaders = {'train':train_dataloader, 'valid': valid_dataloader}
        self._make_checkpoint_dir()
        
        sys.stdout = open(self.model_checkpoint_dir +'/training_log.txt', 'w')
        print(train_dataloader.dataset.transforms)

        best_score = 0
        for epoch in tqdm(range(self.config['num_epochs'])):
            if self.config['model']['output_structure'] == 'single':
                best_score = self.train_and_validate_one_epoch(model, epoch, dataloaders, 
                                                            criterion_ce, optimizer, best_score)

            elif self.config['model']['output_structure'] == 'multiple':
                best_score = self.train_and_validate_one_epoch_multiple(model, epoch, dataloaders, 
                                                            criterion_ce, criterion_bce, optimizer, best_score)
            
        sys.stdout.close()

    def _checkpoint(self, model, epoch, f1score):
        state = {
            'model' : model.state_dict(),
            'epoch' : epoch,
            'f1_score' : f1score
        }
        torch.save(state, os.path.join(self.model_checkpoint_dir, f"{self.config['model']['model_name']}.pt"))

    def _make_checkpoint_dir(self):
        if not os.path.exists(self.model_checkpoint_dir):
            os.makedirs(self.model_checkpoint_dir)

        shutil.copy('./config/config.yaml', self.model_checkpoint_dir)

def set_randomseed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
        

            
            
