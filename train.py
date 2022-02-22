from train.trainer import Trainer
from utils.dataloader import TrainLoaderWrapper
from config.config import config 
import pandas as pd
import os 
import warnings
warnings.filterwarnings('ignore')

def main():
    train_df = pd.read_csv(os.path.join(config['input_dir'], 'train_meta.csv')) 
    if config['debuging']:
        train_df = train_df.loc[:1000]
        print('debugging mode')

    feeder = TrainLoaderWrapper(config, train_df)
    trainer = Trainer(config, feeder)
    trainer.train()

if __name__ == '__main__':
    main()