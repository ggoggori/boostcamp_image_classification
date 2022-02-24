from utils.dataloader import TestLoaderWrapper
from utils.utils import get_device
from models.model import Network
import torch
import yaml
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    config = yaml.load(open("./config/" + "config.yaml", "r"), Loader=yaml.FullLoader)
    info = pd.read_csv(os.path.join(config['dir']['input_dir'], 'eval/info.csv')) 
    feeder = TestLoaderWrapper(config, info)
    dataloader = feeder.make_dataloader()

    device = get_device()
    model = Network()
    model.load_state_dict(torch.load('checkpoint/23d-15h-37m/resnet-18.pth')['model'])
    model = model.to(device)
    
    preds = []
    for image in dataloader:
        image = image.to(device)
        pred = model(image) 
        pred = pred.argmax(dim=-1)
        preds.extend(pred.cpu().numpy())
    
    info['ans'] = preds
    info.to_csv('submission.csv', index=False)
    print('inference is done!')

if __name__ == '__main__':
    main()
