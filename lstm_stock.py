
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta

# ------------------- Streamlit UI -------------------
st.title("📈 LSTM Stock Price Prediction")

# Sidebar options
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g.MSTR):", "MSTR")
start_date = st.sidebar.date_input("Start Date", dt.date(2024, 6, 1))
end_date = st.sidebar.date_input("End Date", dt.date(2025, 1, 31))
future_days = st.sidebar.slider("Select Future Prediction Days:", min_value=1, max_value=30, value=7)

# ------------------- Load Stock Data -------------------
st.subheader(f"📊 Fetching {stock_symbol} Stock Data...")
data = yf.download(stock_symbol, start=start_date, end=end_date)

if data.empty:
    st.error("❌ No data found. Please check the stock symbol or date range.")
    st.stop()

data = data[['Close']]
st.write("📜 Last 5 rows of dataset:")
st.write(data.tail())

# ------------------- Data Preprocessing -------------------
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

x_train, y_train = [], []
time_step = 10  # Number of past days used for prediction

for i in range(time_step, len(data_scaled)):
    x_train.append(data_scaled[i-time_step:i])
    y_train.append(data_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# ------------------- LSTM Model -------------------
st.subheader("🔍 Training LSTM Model")

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(units=50),
    Dense(units=1)
])

model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)

with st.spinner("Training the LSTM model..."):
    model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=0, shuffle=False, callbacks=[early_stop])
st.success("✅ Model Training Completed!")

# ------------------- Prediction on Test Data -------------------
st.subheader("📡 Testing the Model")

# Ensure enough data is available for testing
if len(data_scaled) > time_step:
    test_data = data_scaled[-time_step:]
    x_test = np.array(test_data).reshape(1, time_step, 1)
else:
    st.error("❌ Not enough data to create test samples. Try selecting a longer date range.")
    st.stop()

# Predict
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# ------------------- Future Predictions -------------------
st.subheader(f"📅 Predicting Next {future_days} Days")

future_predictions = []
last_data = test_data.reshape(1, time_step, 1)

for _ in range(future_days):
    future_pred = model.predict(last_data)
    future_predictions.append(future_pred[0, 0])
    
    # Shift the window for the next prediction
    last_data = np.roll(last_data, shift=-1, axis=1)
    last_data[0, -1, 0] = future_pred

# Convert future predictions back to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate future dates
curr_date = data.index[-1]
future_dates = [curr_date + timedelta(days=i+1) for i in range(future_days)]

# Create DataFrame for future predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predictions': future_predictions.flatten()})
future_df.set_index('Date', inplace=True)

st.write("📌 **Future Predictions:**")
st.write(future_df)

# ------------------- Model Evaluation -------------------
if len(data) >= future_days:
    true_values = data.iloc[-future_days:]['Close'].values
    predicted_values = future_predictions.flatten()[:len(true_values)]

    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    r2 = r2_score(true_values, predicted_values)

    st.subheader("📊 Model Performance")
    st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
    st.write(f"**R² Score:** {r2:.4f}")

st.success("✅ Prediction Completed!")
