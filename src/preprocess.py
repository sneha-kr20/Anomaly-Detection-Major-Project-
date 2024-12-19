import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):

    # Drop unnecessary columns
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    df['datatime'] = pd.to_datetime(df['datatime'])
    df.set_index('datatime', inplace=True)
    df_resampled = df.resample('T').mean()
    df_resampled.dropna(inplace=True)
    df_resampled.reset_index(inplace=True)

    # Handle missing values
    df_resampled.fillna(df_resampled.mean(), inplace=True)

    # Remove duplicates
    df_resampled = df_resampled[~df_resampled["datatime"].duplicated(keep="last")].reset_index(drop=True)

    # Scale columns
    cols_to_scale = ['temperature', 'humidity', 'mq2_analog', 'mq9_analog', 
                     'mq8_analog', 'pm25_density', 'pm10_density', 'sound_analog']
    
    def check_negative_or_zero(row):
        for col in cols_to_scale:
            if row[col] <= 0:
                return -1 
        return 1 

    # Apply the check function to create the 'manual_anomaly' column
    df_resampled['manual_anomaly'] = df_resampled.apply(check_negative_or_zero, axis=1)
    scaler = StandardScaler()
    df_resampled[cols_to_scale] = scaler.fit_transform(df_resampled[cols_to_scale])
    print(df_resampled.head(5))
    return df_resampled
