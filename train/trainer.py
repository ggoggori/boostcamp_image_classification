import torch
from models.model import Network
from utils.metrics import cal_metric
from tqdm import tqdm
import os
import time
import shutil

## to-do
'''
checkpoint 추가
config로 변수 바꿔주기
다중 아웃풋 모델 고려
criterion, optimizer 두개 config로 관리하기
'''
class Trainer(object):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.device = self._get_device()
        now = time.localtime()
        date = f'{now.tm_mday}d-{now.tm_hour}h-{now.tm_min}m'
        self.model_checkpoint_dir = os.path.join(self.config['dir']['checkpoint_dir'], date)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'working on {device}')
        return device

    def train_and_validate_one_epoch(self, model, epoch, dataloaders, criterion, optimizer, best_score):
        for phase in ['train', 'valid']:
            dataloader = dataloaders[phase]
            running_loss, correct = 0,0
            targets, predictions = torch.Tensor([]), torch.Tensor([])
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

                targets = torch.cat((targets, target.detach().cpu()))
                predictions = torch.cat((predictions, preds.detach().cpu()))

                running_loss += loss.item() * image.size(0)
                correct += torch.sum(preds==target)
            
            total_loss = running_loss / len(dataloader.dataset)
            f1score, total_acc = cal_metric(predictions, targets, total_size)
            print(f'{phase}-[EPOCH:{epoch}] |F1: {f1score:.3f} | ACC: {total_acc:.3f} | Loss: {total_loss:.5f}|')

            if phase == 'valid' and f1score > best_score:
                best_score = f1score
                print(f'{best_score:.3f} model saved')
                self._checkpoint(model, epoch, best_score)
            
        return best_score
        
    def train(self):
        model = Network().to(self.device)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        
        train_dataloader, valid_dataloader = self.dataset.make_dataloader()
        dataloaders = {'train':train_dataloader, 'valid': valid_dataloader}
        self._make_checkpoint_dir()

        best_score = 0
        for epoch in tqdm(range(self.config['num_epochs'])):
            best_score = self.train_and_validate_one_epoch(model, epoch, dataloaders, 
                                                            criterion, optimizer, best_score)
            
    def _checkpoint(self, model, epoch, f1score):
        state = {
            'model' : model.state_dict(),
            'epoch' : epoch,
            'f1_score' : f1score
        }
        torch.save(state, os.path.join(self.model_checkpoint_dir, f"{self.config['model_name']}.pth"))

    def _make_checkpoint_dir(self):
        if not os.path.exists(self.model_checkpoint_dir):
            os.makedirs(self.model_checkpoint_dir)

        shutil.copy('./config/config.yaml', self.model_checkpoint_dir)



        

            
            
