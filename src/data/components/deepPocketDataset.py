import numpy as np
from ta.trend import cci, dpo, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import pandas as pd
from torch.utils.data import Dataset
import torch
from pandas_ta import hma

class DeepPocketDataset(Dataset):
    def __init__(self, csv_file, window=20, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.window = window


        self.data["Price"] = self.data["Price"].replace(',', '', regex=True).astype(float)
        self.data["High"] = self.data["High"].replace(',', '', regex=True).astype(float)
        self.data["Low"] = self.data["Low"].replace(',', '', regex=True).astype(float)
        self.data["Open"] = self.data["Open"].replace(',', '', regex=True).astype(float)

        # Normalize financial indicators 
        self.data['CCI'] = cci(self.data['High'], self.data['Low'], self.data['Price'], window=self.window)
        self.data['CSI'] = (self.data['Price'] - self.data['Price'].rolling(self.window).min()) / (
                self.data['Price'].rolling(self.window).max() - self.data['Price'].rolling(self.window).min())
        self.data['DI'] = dpo(self.data['Price'], window=self.window)
        self.data['DMI'] = np.log(self.data['Price']) - np.log(self.data['Price'].shift(1))
        self.data['EMA'] = EMAIndicator(self.data['Price'], window=self.window).ema_indicator()
        self.data['HMA'] = hma(self.data['Price'], window=self.window)
        self.data['Momentum'] = RSIIndicator(self.data['Price'], window=self.window).rsi()
        self.data['ATR'] = AverageTrueRange(self.data['High'], self.data['Low'], self.data['Price'],window=self.window).average_true_range()

        # Normalize financial indicators with respect to the closing value on the previous day
        financial_indicators = ['ATR','CCI', 'CSI', 'DI', 'DMI', 'EMA', 'HMA', 'Momentum']
        for indicator in financial_indicators:
            self.data[indicator] = self.data[indicator] / self.data[indicator].shift(1)
        
        # Normalize financial data 
        self.data[['NormalHigh', 'NormalLow', 'NormalPrice']] = self.data[['High', 'Low', 'Price']].div(
            self.data['Open'].rolling(self.window).mean(), axis=0)
        



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # Extract the features and label from the sample
        features = torch.tensor([
            sample['CCI'],
            sample['CSI'],
            sample['DI'],
            sample['DMI'],
            sample['EMA'],
            sample['HMA'],
            sample['Momentum'],
            sample['ATR'],
            sample['NormalHigh'],
            sample['NormalLow'],
            sample['NormalPrice']
        ], dtype=torch.float32)

        label = torch.tensor([
            sample['NormalHigh'],
            sample['NormalLow'],
            sample['NormalPrice']
        ], dtype=torch.float32)

        

        # Apply any transformations if specified
        if self.transform:
            features = self.transform(features)

        return torch.nan_to_num(features), torch.nan_to_num(label)
