import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class BitcoinDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork:
    def __init__(self, input_size, learning_rate=0.001):
        self.model = self._create_model(input_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def _create_model(self, input_size):
        model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        return model
    
    def prepare_data(self, X, y, batch_size=32):
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        y_scaled = pd.Series(y_scaled.flatten(), index=y.index)
        
        return X_scaled, y_scaled
    
    def create_data_loaders(self, X_scaled, y_scaled, train_size, dev_size, batch_size=32):
        # Create dataset
        dataset = BitcoinDataset(X_scaled, y_scaled)
        

        test_size = len(X_scaled) - train_size - dev_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, dev_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader, val_loader, epochs=100):
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch.reshape(-1, 1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = self.model(X_batch)
                    val_loss += self.criterion(y_pred, y_batch.reshape(-1, 1)).item()
            
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        return train_losses, val_losses
    
    def evaluate(self, test_loader):
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = self.model(X_batch)
                predictions.extend(y_pred.numpy().flatten())
                actuals.extend(y_batch.numpy().flatten())
        
        predictions = self.scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
        actuals = self.scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1))
        
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        
        return predictions, actuals, rmse
    
    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = self.scaler_y.inverse_transform(predictions.numpy())
        
        return predictions