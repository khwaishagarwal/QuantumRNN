import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def load_airline_passenger_data(file_path=None):
    """
    Load airline passenger data from a file or a default URL if no file path is provided
    """
    if file_path:
        data = pd.read_csv(file_path)
    else:
        # Load from pandas built-in datasets if file not specified
        data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
    return data

def preprocess_data(data):
    data = data.dropna()  # Remove missing values
    data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Month', inplace=True)
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Passengers']])
    
    return data_scaled, scaler

def split_data(data, train_size=0.8):
    train_size = int(len(data) * train_size)
    train, test = data[:train_size], data[train_size:]
    return train, test

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

class AirlineDataset(Dataset):
    """Dataset for Airline Passengers time series"""
    
    def __init__(self, data, seq_length, train=True, train_ratio=0.8):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Split into train and test
        train_size = int(len(scaled_data) * train_ratio)
        if train:
            self.data = scaled_data[:train_size]
        else:
            self.data = scaled_data[train_size:]
        
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx:idx+self.seq_length, 0])
        y = torch.FloatTensor(self.data[idx+1:idx+self.seq_length+1, 0])
        
        return x.unsqueeze(1), y.unsqueeze(1)  # Add feature dimension
    
    def inverse_transform(self, data):
        """Convert scaled data back to original scale"""
        return self.scaler.inverse_transform(data)

def prepare_dataloaders(batch_size=32, seq_length=12, train_ratio=0.8):
    """Prepare DataLoaders for training and testing"""
    
    # Load dataset
    df = load_airline_passenger_data()
    
    # Create datasets
    train_dataset = AirlineDataset(df[['Passengers']], seq_length=seq_length, train=True, train_ratio=train_ratio)
    test_dataset = AirlineDataset(df[['Passengers']], seq_length=seq_length, train=False, train_ratio=train_ratio)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset.scaler