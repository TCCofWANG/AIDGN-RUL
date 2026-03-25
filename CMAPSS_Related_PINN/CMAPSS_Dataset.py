import sys
sys.path.append("..")
from torch.utils.data import Dataset

"""
In this file, the three datasets will be processed by the Dataset; 
The output will be used as input to the Experiment_CMaps(gai).py
"""

class CMAPSSData(Dataset):

    def __init__(self, data_x, data_y):
        self.data_x         = data_x
        self.data_y         = data_y
        self.data_channel   = data_x.shape[2]

    def __getitem__(self, index):
        sample_x = self.data_x[index]
        sample_y = self.data_y[index]

        return sample_x, sample_y

    def __len__(self):
        return len(self.data_x)

class CMAPSSData_index(Dataset):

    def __init__(self, data_x, basis_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.basis_x = basis_x
        self.data_channel = data_x.shape[2]

    def __getitem__(self, index):
        sample_x = self.data_x[index]
        basis_x = self.basis_x[index]
        sample_y = self.data_y[index]

        return sample_x, basis_x, sample_y

    def __len__(self):
        return len(self.data_x)



class CMAPSSData_index_PINN(Dataset):

    def __init__(self, data_x, basis_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.basis_x = basis_x

    def __getitem__(self, index):
        sample_x = self.data_x[index]
        basis_x = self.basis_x[index]
        sample_y = self.data_y[index]

        return sample_x, basis_x, sample_y

    def __len__(self):
        return len(self.data_x)
