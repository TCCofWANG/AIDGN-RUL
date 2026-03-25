import sys
sys.path.append("..")
from torch.utils.data import Dataset

"""
In this file, the three datasets will be processed by the Dataset; 
The output will be used as input to the Experiment_CMaps(gai).py
"""

class New_CMAPSSData(Dataset):

    def __init__(self, data_x, data_y, Rul):
        self.data_x         = data_x
        self.data_y         = data_y
        self.rul = Rul

    def __getitem__(self, index):
        sample_x = self.data_x[index]
        sample_y = self.data_y[index]
        rul = self.rul[index]

        return sample_x, sample_y, rul

    def __len__(self):
        return len(self.data_x)