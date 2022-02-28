#from sklearn.metrics import f1_score
from torchmetrics.functional import f1_score
from sklearn.metrics import confusion_matrix
import torch

def cal_metric(preds, targets, total_size):
    preds = preds.cpu()
    targets = targets.cpu()
    f1 = f1_score(preds, targets, average='macro', num_classes=18)

    correct = torch.sum(preds==targets)
    acc = float(correct/total_size)

    return f1, acc
    
