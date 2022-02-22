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
    def __init__(self, batch_size:int, train_df:pd.DataFrame, valid_size:float):
        self.batch_size = batch_size
        self.train_df = train_df
        self.valid_size = valid_size
    
    def _make_dataset(self) -> torch.utils.data.Dataset:
        '''
        return dataset
        '''
        indices = [i for i in range(len(self.train_df))]
        np.random.shuffle(indices)
        threshold = len(indices) * (1 - self.valid_size)
        
        train_idx, valid_idx = indices[:threshold], indices[threshold:]
        train_data = self.train_df.loc[train_idx]
        valid_data = self.train_df.loc[valid_idx]

        train_T = _augmentation(mode='train')
        valid_T = _augmentation(mode='valid')

        train_dataset = MaskedFaceDataset(train_data, self.folder, transform=train_T)
        valid_dataset = MaskedFaceDataset(valid_data, self.folder, transform=valid_T)
        
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
            transforms.PILToTensor()
        ])

        test_transforms = transforms.Compose([
            transforms.PILToTensor()
        ])

        return train_transforms, test_transforms
                

        
    
        







