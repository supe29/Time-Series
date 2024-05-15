# Fault detection application with streamlit

Fast API for backend

Streamlit for frontend

Fault detection on sensor data by using STL Decomposition, Isolation forrest and SVM

![image](https://user-images.githubusercontent.com/69507898/196674670-a68bf5c1-80be-49e1-b264-e65d918cf61f.png)
![image](https://user-images.githubusercontent.com/69507898/196674836-13512dc2-7955-4424-ba8d-d8f11a3fa16b.png)

# Project Overview
This project focuses on detecting faults in sensor data by leveraging three powerful techniques: STL (Seasonal-Trend decomposition using Loess) decomposition, Isolation Forest, and Support Vector Machine (SVM). Each method provides a unique approach to anomaly detection, ensuring robust identification of faults. This project involves the extraction, transformation, decomposition, and analysis of sensor data to detect faults using these techniques.

# Data Extraction:
1) Gather sensor data from relevant sources such as IoT devices, industrial sensors, or other data collection systems.
2) Ensure data is collected at regular intervals to maintain the integrity of the time series.

# Data Preprocessing:
1) Clean the data to handle missing values, outliers, and noise.
2) Normalize or standardize the data if required to ensure consistency in analysis.

# STL Decomposition:
1) Apply STL decomposition to the time series data.
2) Decompose the data into seasonal, trend, and residual components.

# Anomaly Detection using STL Residuals:
1) Analyze the residual component to detect anomalies or faults.
2) Set thresholds or use statistical methods to identify significant deviations from the normal pattern.

# Anomaly Detection using Isolation Forest:
1) Train an Isolation Forest model on the sensor data.
2) Identify anomalies based on the Isolation Forest's anomaly scores.

# Anomaly Detection using SVM:
1) Train a One-Class SVM model on the sensor data.
2) Detect anomalies by identifying data points that fall outside the learned decision boundary.

# Visualization and Reporting:
1) Visualize the original data, decomposed components, and detected faults from each method.
2) Generate reports summarizing the findings and highlighting periods with detected faults.

# Technologies Used
Python: For data processing, anomaly detection, and analysis.

Pandas: For data manipulation and cleaning.

Statsmodels: For applying STL decomposition.

Scikit-learn: For implementing Isolation Forest and SVM models.

Matplotlib/Seaborn: For data visualization.

Jupyter Notebook: For interactive analysis and reporting.
