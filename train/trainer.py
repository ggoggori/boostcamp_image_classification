import torch
from models.model import Network
from utils.metrics import cal_metric
from tqdm import tqdm


## to-do
'''
metric f1 추가하기
config로 변수 바꿔주기
다중 아웃풋 모델 고려
criterion, optimizer 두개 config로 관리하기
'''

class Trainer(object):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.device = self._get_device()

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'working on {device}')
        return device

    def train_one_epoch(self, model, dataloader, criterion, optimizer):
        model.train()
        running_loss, correct = 0,0
        targets, predictions = torch.Tensor([]), torch.Tensor([])
        total_size = len(dataloader.dataset)
        for batch in tqdm(dataloader):
            image = batch['image'].to(self.device)
            target = batch['labels']['label'].to(self.device)
            output = model(image)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(output, 1)

            targets = torch.cat((targets, target.detach().cpu()))
            predictions = torch.cat((predictions, preds.detach().cpu()))

            running_loss += loss.item() * image.size(0)
            correct += torch.sum(preds==target)
            
        total_loss = running_loss / len(dataloader.dataset)
        f1score, total_acc = cal_metric(predictions, targets, total_size)

        return total_loss, total_acc, f1score
    
    def _validate(self, model, dataloader, criterion):
        model.eval()
        valid_loss, valid_correct = 0, 0

        with torch.no_grad():
            for batch in dataloader:
                image = batch['image'].to(self.device)
                target = batch['labels']['label'].to(self.device)
                output = model(image)
                loss = criterion(output, target)

                _, preds = torch.max(output, 1) 
                
                valid_loss += loss.item() * image.size(0)
                valid_correct += torch.sum(preds==target)
        
        total_valid_loss = valid_loss/len(dataloader.dataset)
        total_valid_acc = valid_correct/len(dataloader.dataset)

        return total_valid_loss, total_valid_acc
        
    def train(self):
        model = Network().to(self.device)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        train_dataloader, valid_dataloader = self.dataset.make_dataloader()

        for epoch in tqdm(range(self.config['num_epochs'])):
            train_loss, train_metric = self.train_one_epoch(model, train_dataloader, 
                                                        criterion, optimizer)
            print(f'Train-[EPOCH:{epoch}] | ACC: {train_metric:.3f} | Loss: {train_loss:.5f}|')
            valid_loss, valid_metric = self._validate(model, valid_dataloader, 
                                                          criterion)
            print(f'Valid-[EPOCH:{epoch}] | ACC: {valid_metric:.3f} | Loss: {valid_loss:.3f}|')
