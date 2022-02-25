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
    checkpoint_dir = './checkpoint/25d-12h-19m'
    config = yaml.load(open(checkpoint_dir + "/config.yaml", "r"), Loader=yaml.FullLoader)
    info = pd.read_csv(os.path.join(config['dir']['input_dir'], 'eval/info.csv')) 
    feeder = TestLoaderWrapper(config, info)
    dataloader = feeder.make_dataloader()

    device = get_device()
    model = Network(config)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, config['model']['model_name']+'.pt'))['model'])
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
