import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

# Function to apply EWA and calculate RMSE for different alpha values
def test_ewa(data, alphas):
    rmse_values = []
    for alpha in alphas:
        srtt = data[0]
        predictions = [srtt]
        for rtt in data[1:]:
            srtt = (1 - alpha) * srtt + alpha * rtt
            predictions.append(srtt)
        rmse = calculate_rmse(data, predictions)
        rmse_values.append(rmse)
    return rmse_values

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

# Define alpha values to test
alphas = np.linspace(0.1, 0.9, 9)

# Calculate RMSE for each alpha value
rmse_vimeo = test_ewa(data_v, alphas)
rmse_youtube = test_ewa(data_y, alphas)
rmse_reddit = test_ewa(data_r, alphas)

# Plot RMSE vs. Alpha
plt.figure(figsize=(10, 6))
plt.plot(alphas, rmse_vimeo, label='Vimeo')
plt.plot(alphas, rmse_youtube, label='YouTube')
plt.plot(alphas, rmse_reddit, label='Reddit')
plt.xlabel('Weight of New Sample (Alpha)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('RMSE vs. Alpha for EWA')
plt.legend()
plt.grid(True)
plt.show()

