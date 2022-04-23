from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
class BikeSharingDataset(Dataset):
    def __init__(self,csv_path):
        df=pd.read_csv(csv_path,header=None,skiprows=1)
