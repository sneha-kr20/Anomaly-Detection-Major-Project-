import pandas as pd
import numpy as np
from lstm import train_lstm, detect_anomalies_with_lstm
from isolation import detect_anomalies_with_isolation_forest
from feature_engineering import scale_features, calculate_statistical_features, calculate_correlations, extract_time_features, group_by_time
from preprocess import preprocess_data

def detect_anomalies(df):
    
    # Calculate statistical features
    stat_features = calculate_statistical_features(df)

    # Specify columns to scale
    columns_to_scale = ['temperature', 'humidity', 'mq2_analog', 
                        'sound_analog', 'mq9_analog', 
                        'mq8_analog', 'pm25_density', 'pm10_density']

    # Scale selected columns
    df_scaled = scale_features(df, columns_to_scale)

    # Calculate correlations
    high_corr = calculate_correlations(df_scaled)

    # Extract time features (e.g., day, hour, minute)
    df_with_time = extract_time_features(df)

    # Group by time (day, hour, minute)
    grouped_data = group_by_time(df_with_time, ['day', 'hour', 'minute'])
    
    # Step 3: Anomaly detection
    # Apply Isolation Forest anomaly detection
    df_scaled = detect_anomalies_with_isolation_forest(df_scaled)

    # Apply LSTM-based anomaly detection
    df_scaled = detect_anomalies_with_lstm(df_scaled)

    # Combine results of both methods
    df_scaled['combined_anomaly'] = np.where(
        (df_scaled['anomaly_lstm'] == -1) | (df_scaled['anomaly_isolation_forest'] == -1), -1, 1
    )

    return df_scaled

