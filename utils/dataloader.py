import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import os
import numpy as np
import pandas as pd

class MaskedFaceDataset(Dataset):
    def __init__(self, csv_file, folder, transforms=None):
        self.meta = csv_file
        self.transforms = transforms
        self.folder = folder
        
    def __len__(self):
        return len(self.meta)
        
    def __getitem__(self, idx):
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
    def __init__(self, transforms=None):
        pass
        
    def __len__(self):
        pass
        
    def __getitem__(self, idx):
        pass


class TrainLoaderWrapper(object):
    def __init__(self, config, train_df:pd.DataFrame):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.valid_size = self.config['valid_size']
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
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size)

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
                

        
    
        







