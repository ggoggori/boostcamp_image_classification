from utils.dataloader import TestLoaderWrapper
from utils.utils import get_device
from models.model import Network
import torch
import yaml
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


def get_class(gender, age, mask):
    a = (gender >= 0.5).float().squeeze()
    b = age.argmax(dim=1)
    c = mask.argmax(dim=1)
    d = torch.stack((a,b,c))
    return [(i[0]*3 + i[1] + i[2]*6).item() for i in d.T]

def main():
    checkpoint_dir = './checkpoint/26d-16h-51m'
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
        if config['model']['output_structure'] == 'single':
            pred = model(image) 
            pred = pred.argmax(dim=-1)
            pred = pred.cpu().numpy()

        elif config['model']['output_structure'] == 'multiple':
            gender, age, mask = model(image)
            pred = get_class(gender, age, mask)
            
        preds.extend(pred)

    info['ans'] = preds
    info.to_csv('submission.csv', index=False)
    print('inference is done!')

if __name__ == '__main__':
    main()
