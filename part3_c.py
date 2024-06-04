from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from part3_a import calculate_rmse

# Load RTT data
data_v = pd.read_csv('Vimeo.csv')['RTT']
data_y = pd.read_csv('Youtube.csv')['RTT']
data_r = pd.read_csv('Reddit.csv')['RTT']
# Find the length of the shortest dataframe
min_length = min(len(data_v), len(data_y), len(data_r))

# Trim the dataframes to the length of the shortest dataframe
data_v = data_v[:min_length].values
data_y = data_y[:min_length].values
data_r = data_r[:min_length].values

# Prepare data for linear regression
X = np.arange(len(data_v)).reshape(-1, 1)
y = data_v

# Train the model
model = LinearRegression().fit(X, y)

# Predict RTT
predictions_lr = model.predict(X)

# Calculate RMSE
rmse_lr = calculate_rmse(y, predictions_lr)

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(y, label='Actual RTT')
plt.plot(predictions_lr, label='Linear Regression Prediction')
plt.xlabel('Sample Index')
plt.ylabel('RTT (seconds)')
plt.title('RTT Prediction Using Linear Regression (Vimeo)')
plt.legend()
plt.grid(True)
plt.show()

print(f'RMSE for Linear Regression: {rmse_lr:.4f}')
