import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader,Dataset,random_split
from multiprocessing import cpu_count

class loader(pl.LightningDataModule):
    def __init__(self,ptname,batch_size=256):
        super().__init__()
        self.batch_size=batch_size
        self.data=torch.load(ptname).long()
        self.len=len(self.data)
        self.train,self.val,self.test=None,None,None
        self.split()
    def split(self,ratio=(0.8,0.1,0.1)):
        l = [int(i * self.len) for i in ratio]
        l[-1] += self.len - sum(l)
        self.train,self.val,self.test=random_split(self.data,l)

    def prepare_data(self) -> None:
        ...

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=cpu_count())

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=cpu_count())
