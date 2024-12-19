from sklearn.ensemble import IsolationForest

def detect_anomalies_with_isolation_forest(df, contamination=0.05):
    # Select numeric columns for anomaly detection
    numeric_data = df.select_dtypes(include=['float64', 'int64'])
    
    # Initialize and fit IsolationForest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(numeric_data)
    
    # Predict anomalies (-1 for anomaly, 1 for normal)
    df['anomaly_isolation_forest'] = iso_forest.predict(numeric_data)
    
    # Override with manual_anomaly if it is -1
    df.loc[df['manual_anomaly'] == -1, 'anomaly_isolation_forest'] = -1
    
    return df
