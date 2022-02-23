import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import os
import numpy as np
import pandas as pd

class MaskedFaceDataset(Dataset):
    def __init__(self, csv_file:pd.DataFrame, folder:str, transforms=None):
        self.meta = csv_file
        self.transforms = transforms
        self.folder = folder
        
    def __len__(self):
        return len(self.meta)
        
    def __getitem__(self, idx:int) -> dict:
        sample = self.meta.loc[idx]
        gender = sample['gender_label']
        age = sample['age_label']
        mask = sample['mask_label']
        label = sample['class']
        image = Image.open(os.path.join(self.folder,sample['detail_path']))
        if self.transforms is not None:
            image = self.transforms(image)
        
        sample = {'image':image.float(), 'labels':{'gender':gender,
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

        return image.float()

class TrainLoaderWrapper(object):
    def __init__(self, config:dict, train_df:pd.DataFrame):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.valid_size = self.config['valid_size']
        self.num_workers = self.config['num_workers']
        self.train_df = train_df
        
    def _make_dataset(self) -> torch.utils.data.Dataset:
        '''
        return dataset
        '''
        indices = [i for i in range(len(self.train_df))]
        np.random.shuffle(indices)
        threshold = int(len(indices) * (1 - self.valid_size))
        train_idx, valid_idx = indices[:threshold], indices[threshold:]
        train_data = self.train_df.loc[train_idx].reset_index(drop=True)
        valid_data = self.train_df.loc[valid_idx].reset_index(drop=True)

        train_T, valid_T = self._augmentation()

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
    
    def _augmentation(self) -> torchvision.transforms:
        '''
        return torchvision.transforms
        '''

        train_transforms = transforms.Compose([
            transforms.CenterCrop((350)),
            transforms.Resize((224,224)),
            transforms.PILToTensor()
        ])

        test_transforms = transforms.Compose([
            transforms.CenterCrop((350)),
            transforms.Resize((224,224)),
            transforms.PILToTensor()
        ])

        return train_transforms, test_transforms
                
class TestLoaderWrapper():
    def __init__(self, config:dict, info:pd.DataFrame):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.num_workers = self.config['num_workers']
        self.info = info
    
    def _make_dataset(self) -> torch.utils.data.Dataset:
        test_T = self._augmentation()
        dataset = MaskedFaceDataset_Test(self.info, self.config['dir']['image_dir'].format('eval'),
                                                transforms=test_T)

        return dataset

    def make_dataloader(self) -> torch.utils.data.DataLoader:
        test_dataset = self._make_dataset()
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                                    num_workers=self.num_workers, shuffle=False)

        return test_dataloader

    def _augmentation(self) -> torchvision.transforms:
        test_transforms = transforms.Compose([
            transforms.CenterCrop((350)),
            transforms.Resize((224,224)),
            transforms.PILToTensor()
        ])

        return test_transforms

        
    
        







