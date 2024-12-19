import pandas as pd
from classification import run_classification  
from anomaly_detection import detect_anomalies 
from preprocess import preprocess_data  
from visualization import visualize_anomalies
import os

def main():
    # Step 1: Load and preprocess data
    file_path = '../data/raw/dht11_new.csv'  
    df_raw = pd.read_csv(file_path)

    # Preprocess the raw data
    df_preprocessed = preprocess_data(df_raw)

    # Step 2: Run anomaly detection (if applicable)
    df_anomaly_detected = detect_anomalies(df_preprocessed)

    # Save the anomaly-detected DataFrame
    anomaly_detected_file_path = '../data/processed/anomaly_detected.csv'
    df_anomaly_detected.to_csv(anomaly_detected_file_path, index=False)
    print(f"Anomaly-detected data saved to {anomaly_detected_file_path}")


    # Step 3: Run classification on the anomaly detected data
    run_classification(df_anomaly_detected)  

    # Step 4: Visualize anomalies using the preprocessed file (ensure the path is correct)
    processed_file_path = '../data/processed/classification_dht11_preprocessed.csv'

    # Check if the processed file exists before visualization
    if os.path.exists(processed_file_path):
        visualize_anomalies(processed_file_path)
    else:
        print(f"Processed data file {processed_file_path} not found!")

    # Step 5: Completion message
    print("Classification and anomaly detection completed successfully.")

# Call the main function
if __name__ == "__main__":
    main()
