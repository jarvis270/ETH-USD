import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Download historical data for Ethereum
eth = yf.Ticker("ETH-USD")
hist = eth.history(period="1y")[['Close']]  # Use 1 year of data for better accuracy

# Prepare the data
hist['Date'] = hist.index
hist['Date'] = pd.to_datetime(hist['Date'])
hist['Date'] = hist['Date'].map(pd.Timestamp.toordinal)
X = hist[['Date']]
y = hist['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R-squared (Confidence Score): {r2}")

# Example prediction
new_day = pd.Timestamp('2024-11-11').toordinal()  # Convert date to ordinal
predicted_price = model.predict([[new_day]])
print(f"Predicted Price for 2024-11-11: {predicted_price[0]}")
