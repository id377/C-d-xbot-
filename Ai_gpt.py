Python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the network traffic dataset from a CSV file
df = pd.read_csv('network_traffic.csv')

# Display the first few rows of the data to understand its structure (optional)
print("Data preview:")
print(df.head())

# Feature selection: Use 'packet_size' and 'duration' for anomaly detection
X = df[['packet_size', 'duration']]

... # Train-test split: 80% training, 20% testing
... X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
... 
... # Initialize the Isolation Forest model
... iso_forest = IsolationForest(contamination=0.05, random_state=42)
... 
... # Train the model on the training data
... iso_forest.fit(X_train)
... 
... # Predict anomalies on the test data
... y_pred = iso_forest.predict(X_test)
... 
... # Mark anomalies: -1 = Anomaly, 1 = Normal
... X_test['anomaly'] = np.where(y_pred == -1, 'Anomaly', 'Normal')
... 
... # Print the number of anomalies detected
... num_anomalies = len(X_test[X_test['anomaly'] == 'Anomaly'])
... print(f"Anomalies detected: {num_anomalies}")
... 
... # Show the first few rows of the test data with anomaly labels (optional)
... print("Test data with anomaly labels:")
... print(X_test.head())
... 
... # Visualize the anomalies in a scatter plot
... plt.figure(figsize=(10, 6))
... plt.scatter(X_test['packet_size'], X_test['duration'], 
...             c=(X_test['anomaly'] == 'Anomaly'), cmap='coolwarm', label='Anomalies')
... plt.xlabel('Packet Size')
... plt.ylabel('Duration')
... plt.title('Network Traffic Anomalies Detected')
... plt.legend()
... plt.show()
... 
... # ----> NEW CODE ADDED HERE <----
... 
... # Save anomalies to a CSV file
... anomalies = X_test[X_test['anomaly'] == 'Anomaly']
... anomalies.to_csv('detected_anomalies.csv', index=False)
