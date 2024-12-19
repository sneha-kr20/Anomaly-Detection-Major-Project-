import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_anomalies(file_path):
    # Create the 'plot' directory if it does not exist
    if not os.path.exists('plot'):
        os.makedirs('plot')

    # Load the preprocessed data
    df = pd.read_csv(file_path)
    df['datatime'] = pd.to_datetime(df['datatime'])
    df_last_70 = df.tail(int(len(df) * 0.7))  # Select the last 70% of the data

    # Check if the necessary column exists
    if 'anomaly_class' not in df_last_70.columns:
        print("Anomaly classification column not found!")
        return

    # Calculate total and percentage of anomalies
    total_data_points = len(df_last_70)
    total_anomalies = len(df_last_70[df_last_70['combined_anomaly'] == -1])
    lstm_anomalies = len(df_last_70[(df_last_70['anomaly_lstm'] == -1)])
    isolation_anomalies = len(df_last_70[(df_last_70['anomaly_isolation_forest'] == -1)])
    overlap_anomalies = len(df_last_70[(df_last_70['anomaly_lstm'] == -1) & (df_last_70['anomaly_isolation_forest'] == -1)])

    # Calculate percentage of anomalies
    total_anomalies_pct = (total_anomalies / total_data_points) * 100
    lstm_anomalies_pct = (lstm_anomalies / total_data_points) * 100
    isolation_anomalies_pct = (isolation_anomalies / total_data_points) * 100
    overlap_anomalies_pct = (overlap_anomalies / total_data_points) * 100

    # Sensor columns to plot
    sensor_columns = ['temperature', 'humidity', 'mq2_analog', 'mq9_analog', 'sound_analog', 'pm25_density', 'pm10_density']

    # Plot anomalies for each sensor column
    for col in sensor_columns:
        plt.figure(figsize=(12, 6))

        # Plot normal data
        sns.lineplot(data=df_last_70[df_last_70['combined_anomaly'] == -1], x='datatime', y=col, label='Total Anomaly', linewidth=2)

        # Plot LSTM anomalies
        sns.scatterplot(data=df_last_70[(df_last_70['anomaly_lstm'] == -1)], 
                        x='datatime', y=col, color='red', label='LSTM Anomaly', s=60, marker='*')

        # Plot Isolation Forest anomalies
        sns.scatterplot(data=df_last_70[(df_last_70['anomaly_isolation_forest'] == -1)], 
                        x='datatime', y=col, color='green', label='Isolation Forest Anomaly', s=60, marker='x')

        # Plot Combined anomalies
        sns.scatterplot(data=df_last_70[(df_last_70['anomaly_lstm'] == -1) & (df_last_70['anomaly_isolation_forest'] == -1)], 
                        x='datatime', y=col, color='blue', label='Common Anomaly', s=60, marker='D')

        # Add title, labels, and grid
        plt.title(f'{col} with Anomalies (Percentage: {total_anomalies_pct:.2f}%)', fontsize=16, fontweight='bold')
        plt.xlabel('Datetime', fontsize=14)
        plt.ylabel(col.capitalize(), fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'plot/{col}_anomaly_plot.png')
        plt.close()

    # Anomaly percentages for comparison
    anomaly_percentages = {
        'LSTM Anomalies': lstm_anomalies_pct,
        'Isolation Forest Anomalies': isolation_anomalies_pct,
        'Combined Anomalies': overlap_anomalies_pct,
    }

    # Plot bar chart of anomaly percentages
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(anomaly_percentages.keys()), y=list(anomaly_percentages.values()), palette='viridis')
    plt.title('Anomaly Detection Method Comparison (in %)', fontsize=16, fontweight='bold')
    plt.ylabel('Percentage of Anomalies', fontsize=14)
    plt.tight_layout()

    # Save the anomaly comparison plot
    plt.savefig('plot/anomaly_comparison_bar_plot.png')
    plt.close()
