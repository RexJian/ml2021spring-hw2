import numpy as np
import torch
from torch.utils.data import Dataset


class PhonemeDataset(Dataset):

    def __init__(self,x,y=None):
        if(y is None):
            self.y=y
        else:
            y.squeeze(1)
            self.y=torch.LongTensor(np.array(y,dtype=np.int))
        self.x=torch.Tensor(np.array(x,dtype=np.float))
    def __getitem__(self, item):
        if self.y is None:
            return self.x[item]
        else:
            return self.x[item],self.y[item]
    def __len__(self):
        return len(self.x)
