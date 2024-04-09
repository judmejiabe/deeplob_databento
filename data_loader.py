import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

def one_hot_encode(x):
    if x <= -1:
        # Represents the lower threshold
        return [1, 0, 0]  
    elif x >= 1:
        # Represents the upper threshold
        return [0, 0, 1]  
    else:
        # Represents the values between the thresholds
        return [0, 1, 0]  

def preprocess_data(path):

    df = pd.read_csv(path, index_col = 'timestamp')

    target = pd.DataFrame({'mp' : (df['ASKp1'] + df['BIDp1']) / 2})
    target['m_minus_10'] = target['mp'].rolling(window= 11).mean()
    target['m_plus_10'] = target.iloc[::-1]['mp'].rolling(window= 10).\
        mean().iloc[::-1].shift(-1)
    target['diff_10'] = target['m_plus_10'] - target['m_minus_10']

    target['m_minus_20'] = target['mp'].rolling(window= 21).mean()
    target['m_plus_20'] = target.iloc[::-1]['mp'].rolling(window= 20).\
        mean().iloc[::-1].shift(-1)
    target['diff_20'] = target['m_plus_20'] - target['m_minus_20']

    target['m_minus_50'] = target['mp'].rolling(window= 51).mean()
    target['m_plus_50'] = target.iloc[::-1]['mp'].rolling(window= 50).\
        mean().iloc[::-1].shift(-1)
    target['diff_50'] = target['m_plus_50'] - target['m_minus_50']

    one_hot_10 = target['diff_10'].apply(one_hot_encode)
    target[['t10_low', 't10_stable', 't10_up']] = \
        pd.DataFrame(one_hot_10.tolist(), index=target.index)

    one_hot_20 = target['diff_20'].apply(one_hot_encode)
    target[['t20_low', 't20_stable', 't20_up']] = \
        pd.DataFrame(one_hot_20.tolist(), index=target.index)

    one_hot_50 = target['diff_50'].apply(one_hot_encode)
    target[['t50_low', 't50_stable', 't50_up']] = \
        pd.DataFrame(one_hot_50.tolist(), index=target.index)
    
    #Only include data from 9:40 to 15:50
    ts_start = 34800000000000
    ts_end = 57000000000000

    df = df[(df.index >= ts_start) & (df.index <= ts_end)]
    target = target[(target.index >= ts_start) & (target.index <= ts_end)]

    #Normalization

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df)
    standardized_data = \
        pd.DataFrame(standardized_data, index=df.index, columns=df.columns)

    return standardized_data, target

#Defines dataset
class DeepLOBDataset(Dataset):
    def __init__(self, data, target, window_size):
        self.data = data
        self.target = target
        self.window_size = window_size
    
    def __len__(self):
        return(len(self.data) // self.window_size - 1)

    def __getitem__(self, idx):
        x = torch.Tensor(np.array(
            self.data[idx * self.window_size:
                      (idx * self.window_size + self.window_size)]))
        y = torch.Tensor(np.array(
            self.target[['t10_low', 't10_stable', 't10_up', 
                         't20_low', 't20_stable', 't20_up', 
                         't50_low', 't50_stable', 't50_up']]\
                        [idx * self.window_size:
                         (idx * self.window_size + self.window_size)]))
        return x, y

if __name__ == '__main__':
    
    window_size = 100

    data_path = Path('./data.nosync')
    data_path.mkdir(parents=True, exist_ok=True)
    recorded_data_path = data_path / 'Recorded_Data'
    recorded_data_path.mkdir(parents=True, exist_ok=True)
    standardized_data_path = data_path / 'Standardized_Data'
    standardized_data_path.mkdir(parents=True, exist_ok=True)

    for file in recorded_data_path.iterdir():
        if file.is_file():
            standardized_data, target = preprocess_data(file)
            dataset = DeepLOBDataset(standardized_data, target, window_size)
            # Do something with the dataset

            # Save standardized_data and target
            save_path = standardized_data_path / file.name
            standardized_data.to_csv(save_path)
            target_path = standardized_data_path / f"target_{file.name}"
            target.to_csv(target_path)
