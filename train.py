from train.trainer import Trainer
from utils.dataloader import TrainLoaderWrapper
import pandas as pd
import os 
import yaml
import warnings
warnings.filterwarnings('ignore')

def main():
    config = yaml.load(open("./config/" + "config.yaml", "r"), Loader=yaml.FullLoader)
    train_df = pd.read_csv(os.path.join(config['dir']['input_dir'], 'train_meta.csv')) 
    if config['experiment']['debuging']:
        train_df = train_df.loc[:100]
        print('debugging mode')

    feeder = TrainLoaderWrapper(config, train_df)
    trainer = Trainer(config, feeder)
    trainer.train()

if __name__ == '__main__':
    main()

# def set_randomseed():
#     random_seed = 42
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(random_seed)
#     random.seed(random_seed)