import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def split_dataset(data, test_size=0.2, random_state=None):
    from sklearn.model_selection import train_test_split
    
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def transform_data_for_model(data):
    # Assuming the model requires data in a specific format
    # This function should be customized based on the model's input requirements
    transformed_data = data.reshape(data.shape[0], -1)  # Flatten the data if necessary
    return transformed_data

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Perform data cleaning and preprocessing steps here
    # For example, handling missing values, encoding categorical variables, etc.
    
    return data

def load_airline_data(filepath=None):
    """
    Load Airline Passenger dataset
    
    If filepath is not provided, it uses the default dataset from pandas
    """
    if filepath:
        df = pd.read_csv(filepath)
    else:
        # Load from pandas built-in datasets if file not specified
        df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
    
    # Convert 'Month' column to datetime
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Set 'Month' as index
    df.set_index('Month', inplace=True)
    
    return df

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
    df = load_airline_data()
    
    # Create datasets
    train_dataset = AirlineDataset(df[['Passengers']], seq_length=seq_length, train=True, train_ratio=train_ratio)
    test_dataset = AirlineDataset(df[['Passengers']], seq_length=seq_length, train=False, train_ratio=train_ratio)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset.scaler