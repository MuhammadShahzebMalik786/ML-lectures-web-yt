# 📈 AI Stock Price Prediction - Simple Version
# Run this to train the model, then run app.py for the web interface

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("🚀 Starting AI Stock Prediction...")

# 1. Get Tesla stock data
print("📊 Getting Tesla data...")
tesla = yf.Ticker("TSLA")
data = tesla.history(period="1y")
print(f"Got {len(data)} days of data")

# 2. Create simple features
print("🔧 Creating features...")
data['Day'] = range(len(data))
data['MA_5'] = data['Close'].rolling(5).mean()  # 5-day average
data['MA_10'] = data['Close'].rolling(10).mean()  # 10-day average
data = data.dropna()

# Features to use
features = ['Day', 'Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10']
X = data[features]
y = data['Close']  # Price to predict

print(f"Features: {features}")
print(f"Data shape: {X.shape}")

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train AI model
print("🤖 Training AI model...")
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Check accuracy
accuracy = model.score(X_test, y_test)
mse = mean_squared_error(y_test, predictions)

print(f"\n🎯 Results:")
print(f"Accuracy: {accuracy*100:.1f}%")
print(f"Average error: ${mse**0.5:.2f}")

# 7. Show some predictions
print(f"\n📈 Sample Predictions:")
for i in range(5):
    actual = y_test.iloc[i]
    predicted = predictions[i]
    print(f"Actual: ${actual:.2f} | Predicted: ${predicted:.2f} | Diff: ${abs(actual-predicted):.2f}")

# 8. Predict tomorrow's price
print(f"\n🔮 Tomorrow's Prediction:")
latest_data = X.iloc[-1:].values
tomorrow_price = model.predict(latest_data)[0]
current_price = data['Close'].iloc[-1]

print(f"Current price: ${current_price:.2f}")
print(f"AI prediction: ${tomorrow_price:.2f}")
print(f"Expected change: ${tomorrow_price - current_price:.2f}")

# 9. Simple plot
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:20], label='Actual', marker='o')
plt.plot(predictions[:20], label='AI Predicted', marker='x')
plt.title('Tesla Stock: AI vs Reality (First 20 test samples)')
plt.legend()
plt.ylabel('Price ($)')
plt.xlabel('Sample')
plt.show()

print("\n✅ Done! Now run: streamlit run app.py")
