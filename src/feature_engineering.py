import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def group_by_time(df, time_columns, agg_func='mean'):
    # Grouping by the specified time columns
    grouped_df = df.groupby(time_columns).agg(agg_func)
    
    # If you want to reset the index and make 'time_columns' regular columns
    grouped_df.reset_index(inplace=True)
    
    return grouped_df


def create_sequences(data, time_steps):
    data = np.array(data)  
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


def calculate_statistical_features(df):
    stat_features = df.describe().T[['mean', 'min', 'max', 'std']].reset_index()
    stat_features = stat_features.rename(columns={'index': 'Feature'})
    print("Statistical Features (Mean, Min, Max, Standard Deviation):")
    print(stat_features)
    return stat_features


def scale_features(df, cols_to_scale):
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df


def calculate_correlations(df_scaled):
    correlation_matrix = df_scaled.corr()
    high_corr = correlation_matrix.where((correlation_matrix > 0.3) & (correlation_matrix < 1.0))
    high_corr = high_corr.stack().reset_index()
    high_corr.columns = ["Parameter 1", "Parameter 2", "Correlation"]
    high_corr = high_corr.drop_duplicates(subset=["Parameter 1", "Parameter 2"])
    print("Significant correlations:")
    print(high_corr)
    return high_corr


def extract_time_features(df):
    df['datatime'] = pd.to_datetime(df['datatime'])
    df['day'] = df['datatime'].dt.day
    df['hour'] = df['datatime'].dt.hour
    df['minute'] = df['datatime'].dt.minute
    return df


# Anomaly Duration Calculation
def calculate_anomaly_durations(df):
    df['datatime'] = pd.to_datetime(df['datatime'], unit='s')
    df['anomaly_group'] = (df['combined_anomaly'] != -1).cumsum()
    anomalies = df[df['combined_anomaly'] == -1].groupby('anomaly_group')
    anomaly_durations = anomalies['datatime'].agg(['min', 'max'])
    anomaly_durations['anomaly_duration'] = (anomaly_durations['max'] - anomaly_durations['min']).dt.total_seconds()
    df = df.merge(anomaly_durations[['anomaly_duration']], how='left', left_on='anomaly_group', right_index=True)
    df.loc[df['combined_anomaly'] != -1, 'anomaly_duration'] = 0
    df.drop(columns=['anomaly_group'], inplace=True)
    return df
