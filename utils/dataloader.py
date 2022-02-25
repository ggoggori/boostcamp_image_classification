import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
#from sklearn.model_selection import StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from utils.processing import *
import PIL.Image as Image
import os
import numpy as np
import pandas as pd

class MaskedFaceDataset(Dataset):
    def __init__(self, csv_file:pd.DataFrame, folder:str, transforms=None):
        self.csv_file = csv_file
        self.transforms = transforms
        self.folder = folder

    def __len__(self):
        return len(self.csv_file)
        
    def __getitem__(self, idx:int) -> dict:
        sample = self.csv_file.loc[idx]
        gender = sample['gender_label']
        age = sample['age_label']
        mask = sample['mask_label']
        label = sample['class']
        image = Image.open(os.path.join(self.folder, sample['detail_path']))

        if self.transforms is not None:
            image = self.transforms(image)
        
        sample = {'image':image, 'labels':{'gender':gender,
                                                    'age':age, 'mask':mask,
                                                    'label':label}}
        return sample

class MaskedFaceDataset_Test(Dataset):
    def __init__(self, csv_file:pd.DataFrame, folder:str, transforms=None):
        self.csv_file = csv_file
        self.paths = self.csv_file['ImageID'].values
        self.folder = folder
        self.transforms = transforms
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx:int) -> Image:
        path = os.path.join(self.folder, self.paths[idx])
        image = Image.open(path)

        if self.transforms is not None:
            image = self.transforms(image)

        return image

class TrainLoaderWrapper(object):
    def __init__(self, config:dict, train_df:pd.DataFrame):
        self.config = config
        self.train_df = train_df
        self.batch_size = self.config['batch_size']
        self.valid_size = self.config['valid_size']
        self.num_workers = self.config['num_workers']
        
    def _make_dataset(self) -> torch.utils.data.Dataset:
        '''
        return dataset
        '''
        self.train_df['age_label'] = self.train_df['age'].apply(define_age)

        mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=self.config['random_seed'])
        for train_idx, valid_idx in mskf.split(self.train_df, self.train_df[['gender','age_label']]):
            pass
        
        train_data = processing_df(self.train_df.loc[train_idx].reset_index(drop=True), self.config)
        valid_data = processing_df(self.train_df.loc[valid_idx].reset_index(drop=True), self.config)
        
        train_T, valid_T = get_augmentation('train')

        train_dataset = MaskedFaceDataset(train_data, self.config['dir']['image_dir'].format('train'),
                                             transforms=train_T)
        valid_dataset = MaskedFaceDataset(valid_data, self.config['dir']['image_dir'].format('train'),
                                             transforms=valid_T)
        
        return train_dataset, valid_dataset

    def make_dataloader(self) -> torch.utils.data.DataLoader:
        '''
        return dataloader
        '''
        train_dataset, valid_dataset = self._make_dataset()
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                        num_workers=self.num_workers, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, 
                                        num_workers=self.num_workers, shuffle=False)

        return train_dataloader, valid_dataloader
                
class TestLoaderWrapper():
    def __init__(self, config:dict, info:pd.DataFrame):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.num_workers = self.config['num_workers']
        self.info = info
    
    def _make_dataset(self) -> torch.utils.data.Dataset:
        test_T = get_augmentation('test')
        dataset = MaskedFaceDataset_Test(self.info, self.config['dir']['image_dir'].format('eval'),
                                                transforms=test_T)

        return dataset

    def make_dataloader(self) -> torch.utils.data.DataLoader:
        test_dataset = self._make_dataset()
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                                    num_workers=self.num_workers, shuffle=False)

        return test_dataloader
        
def get_augmentation(mode) -> torchvision.transforms:
    '''
    return torchvision.transforms
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.CenterCrop((380)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transforms = transforms.Compose([
        transforms.CenterCrop((380)),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    if mode == 'train':
        return train_transforms, test_transforms 

    elif mode == 'test':
        return test_transforms
        







