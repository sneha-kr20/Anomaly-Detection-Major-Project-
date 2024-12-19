import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import create_sequences

def train_lstm(X, y, epochs=10, batch_size=32):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(32),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model

def detect_anomalies_with_lstm(df, time_steps=50):
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    
    # Scale the data
    scaled_data = scaler.fit_transform(df[numeric_cols])
    
    # Create sequences for LSTM input
    X, y = create_sequences(scaled_data, time_steps)
    
    # Train LSTM model
    model = train_lstm(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate prediction error (mean squared error)
    prediction_error = np.mean((predictions - y) ** 2, axis=1)
    
    # Set the threshold for anomaly detection (95th percentile of errors)
    threshold = np.percentile(prediction_error, 95)
    
    # Initialize 'anomaly_lstm' column with NaN values
    df['anomaly_lstm'] = np.nan
    
    # Mark anomalies based on LSTM predictions
    df.iloc[time_steps:, df.columns.get_loc('anomaly_lstm')] = np.where(prediction_error > threshold, -1, 1)
    
    # If manual_anomaly is -1, mark as anomaly (override LSTM prediction)
    df.loc[df['manual_anomaly'] == -1, 'anomaly_lstm'] = -1
    
    return df
