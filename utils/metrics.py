from sklearn.metrics import f1_score
import torch

def cal_metric(preds, targets, total_size):
    f1 = f1_score(preds, targets, average='macro')

    correct = torch.sum(preds==targets)
    acc = float(correct/total_size)
    return f1, acc
    
