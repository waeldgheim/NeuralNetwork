# Bitcoin Price Prediction using Neural Networks

## Overview
This project implements a deep learning model to predict Bitcoin prices using historical price data. The model takes into account various technical indicators and market features to forecast future price movements of Bitcoin.

## Problem Statement
Predicting cryptocurrency prices is a challenging task due to their high volatility and dependency on multiple factors. This project aims to:
- Analyze the effectiveness of different technical indicators
- Provide insights into price movement patterns
- Evaluate the model's performance using various metrics

## Dataset
The project uses historical Bitcoin price data stored in "BTC-USD.xlsx". The dataset includes:
- Daily price data (Open, High, Low, Close)
- Trading volume
- Date information

Key features engineered from the raw data:
- 30-day moving average
- Price volatility
- Price momentum

## Model Architecture
The project implements a deep neural network with the following characteristics:

- **Input Layer**: Accepts multiple features (price indicators, technical analysis metrics)
- **Hidden Layers**: Multiple fully connected layers with ReLU activation
- **Output Layer**: Single node for price prediction
- **Loss Function**: Mean Squared Error (MSE)

## Project Structure
```
bitcoin-price-prediction/
│
├── README.md
├── requirements.txt
├── BTC-USD.xlsx
├── NeuralNetwork.py
└── Bitcoin_Price_Prediction.ipynb
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd bitcoin-price-prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure all required files are in place:
   - BTC-USD.xlsx (dataset)
   - NeuralNetwork.py (model class)
   - Bitcoin_Price_Prediction.ipynb (main notebook)

2. Run Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `Bitcoin_Price_Prediction.ipynb` and run all cells
