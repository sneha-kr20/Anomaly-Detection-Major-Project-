import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from feature_engineering import calculate_anomaly_durations  

def classify_with_dynamic_thresholds(row, high_score_threshold, high_corr_threshold, mid_score_threshold, mid_corr_threshold):
    # Flag negative values in sensor data as faults (if present)
    # if any(val < 0 for val in row[['temperature', 'humidity', 'mq2_analog', 'sound_analog', 'mq9_analog', 'pm25_density', 'pm10_density']]):
    #     return "Fault" 
    
    # If combined anomaly is 1, 
    if row['combined_anomaly'] == 1:
        return "Normal"
    
    # Classification based on dynamic thresholds for anomaly data (combined_anomaly == -1)
    if row['combined_anomaly'] == -1:
        if row['weighted_score'] > high_score_threshold and row['correlation_behavior'] > high_corr_threshold:
            return "Environmental Changes"  # High score and high correlation indicates environmental changes
        else:
            return "Fault" 
    return "Normal"

def run_classification(df):
    # Step 1: Feature Engineering
    sensor_columns = ['temperature', 'humidity', 'mq2_analog', 'mq9_analog', 'sound_analog', 'pm25_density', 'pm10_density']
    
    # Calculate differences and rolling averages
    for col in sensor_columns:
        df[f'{col}_diff'] = df[col].diff()
        df[f'{col}_rolling_avg'] = df[col].rolling(window=5).mean()

    # Calculate severity (normalized deviation)
    for col in sensor_columns:
        normal_mean = df[df['combined_anomaly'] == 1][col].mean()
        normal_std = df[df['combined_anomaly'] == 1][col].std()
        df[f'{col}_severity'] = (df[col] - normal_mean) / normal_std

    # Compute pairwise correlations
    corr_matrix = df[sensor_columns].corr()

    df = calculate_anomaly_durations(df)

    # Identify strongly correlated sensor pairs (excluding self-pairs)
    strong_pairs = [(col1, col2) for col1 in sensor_columns for col2 in sensor_columns
                    if corr_matrix.loc[col1, col2] > 0.2 and col1 != col2]

    # Initialize a list to hold correlation scores for each row
    correlation_scores = []

    # Iterate over the strong sensor pairs and compute agreement in changes
    for col1, col2 in strong_pairs:
        # Calculate whether both sensors increase or decrease together
        agreement = ((df[f'{col1}_diff'] > 0) & (df[f'{col2}_diff'] > 0)) | \
                    ((df[f'{col1}_diff'] < 0) & (df[f'{col2}_diff'] < 0))
        
        # Append the boolean agreement as integers (1 for agreement, 0 otherwise)
        correlation_scores.append(agreement.astype(int))

    df['correlation_behavior'] = pd.DataFrame(correlation_scores).sum(axis=0) / len(strong_pairs)

    # Step 4: Weighted Scoring - Assign weights to different features
    weights = {
        'trend_features': 0.3,
        'anomaly_duration': 0.2,
        'severity': 0.3,
        'correlation_behavior': 0.2,
    }

    # Define the features to normalize
    features_to_normalize = {
        'trend_features': [f'{col}_diff' for col in sensor_columns],
        'anomaly_duration': ['anomaly_duration'],
        'severity': [f'{col}_severity' for col in sensor_columns],
        'correlation_behavior': ['correlation_behavior']
    }

    # Initialize scaler
    scaler = StandardScaler()
    # print(df.columns)

    # Normalize each feature group and create new normalized columns
    for feature_group, columns in features_to_normalize.items():
        # Normalize and add '_normalized' suffix
        df[[f'{col}_normalized' for col in columns]] = scaler.fit_transform(df[columns])

    # Calculate weighted score using normalized columns
    df['weighted_score'] = (
        weights['trend_features'] * df[[f'{col}_normalized' for col in features_to_normalize['trend_features']]].mean(axis=1) +
        weights['anomaly_duration'] * df['anomaly_duration_normalized'] +
        weights['severity'] * df[[f'{col}_normalized' for col in features_to_normalize['severity']]].mean(axis=1) +
        weights['correlation_behavior'] * df['correlation_behavior_normalized']
    )

    # Calculate thresholds dynamically for weighted_score
    high_score_threshold = df['weighted_score'].quantile(0.9)  # Top 10% for Environmental Changes
    mid_score_threshold = df['weighted_score'].quantile(0.5)  # Median for Fault

    # Calculate thresholds dynamically for correlation_behavior
    high_corr_threshold = df['correlation_behavior'].quantile(0.9)  # Top 10% for strong correlations
    mid_corr_threshold = df['correlation_behavior'].quantile(0.5)  # Median for moderate correlations

    # Apply the classification function to the dataframe
    df['anomaly_class'] = df.apply(lambda row: classify_with_dynamic_thresholds(row, high_score_threshold, high_corr_threshold, mid_score_threshold, mid_corr_threshold), axis=1)

    # Print the unique values and their counts in the 'anomaly_class' column
    print(df['anomaly_class'].value_counts())



    # Save the classified data to a new CSV
    df.to_csv("../data/processed/classification_dht11_preprocessed.csv", index=False)
