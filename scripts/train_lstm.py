import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import ta  # Technical analysis library

#Load Stock Data
ticker = 'RELIANCE.NS'
raw_data = yf.download(ticker, start='2020-01-01', end='2024-12-31', auto_adjust=True)

if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = [col[0] for col in raw_data.columns]

raw_data = raw_data.reset_index()
raw_data['Date'] = pd.to_datetime(raw_data['Date'])

if 'Volume' not in raw_data.columns:
    raw_data['Volume'] = 0

#Load Sentiment
sentiment_df = pd.read_csv("data/reliance_daily_sentiment.csv")
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

#Merge
df = pd.merge(raw_data, sentiment_df, on='Date', how='left')
df['Sentiment'].fillna(0, inplace=True)
df['Sentiment'] = df['Sentiment'].rolling(window=3).mean().fillna(0)

#Add Technical Indicators
df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
df['MACD'] = ta.trend.macd(df['Close'])
df['BB_width'] = ta.volatility.bollinger_wband(df['Close'])
df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

#Drop rows with NaNs from indicators
df.dropna(inplace=True)

#Target: 3-day forward change > 2%
df['Target'] = ((df['Close'].shift(-3) - df['Close']) / df['Close'] > 0.02).astype(int)
df.dropna(inplace=True)

#Features & Scaling
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment',
            'SMA_10', 'RSI', 'MACD', 'BB_width', 'OBV']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

#Sequence Creation
seq_len = 60
X, y = [], []
for i in range(seq_len, len(scaled)):
    X.append(scaled[i-seq_len:i])
    y.append(df['Target'].iloc[i])

X = np.array(X)
y = np.array(y)

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

#LSTM Model with Dropout
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(self.relu(self.fc1(x)))
        return self.out(x)

model = StockLSTM(input_size=X.shape[2])
opt = torch.optim.Adam(model.parameters(), lr=0.001)

#Handle Class Imbalance
neg, pos = np.bincount(y.astype(int))
pos_weight = torch.tensor([neg / pos])
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#Training Loop
for epoch in range(30):
    model.train()
    running_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {running_loss / len(train_loader):.4f}")

#Evaluation
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits).squeeze()
    threshold = 0.55
    preds = (probs > threshold).int()
    y_true = y_test.squeeze().int()

# Metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, preds))
print("\nClassification Report:")
print(classification_report(y_true, preds, digits=4))
print(f"F1 Score: {f1_score(y_true, preds):.4f}")

#Plot
plt.figure(figsize=(12, 6))
plt.plot(probs[:100].numpy(), label='Predicted (Raw)')
plt.plot(y_true[:100].numpy(), label='Actual', alpha=0.7)
plt.title('Stock Trend Prediction (First 100 Samples)')
plt.xlabel('Sample')
plt.ylabel('Trend (0 = Down, 1 = Up)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("imgs/prediction_plot.png", dpi=300)
plt.show()
