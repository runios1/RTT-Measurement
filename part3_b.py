import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from part3_a import test_ewa

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

# Define change point and factors
change_point = min_length // 2
factor_half = 0.5
factor_double = 2

# Apply the change to the data with factor of half
data_v_half = data_v.copy()
data_y_half = data_y.copy()
data_r_half = data_r.copy()
data_v_half[change_point:] *= factor_half
data_y_half[change_point:] *= factor_half
data_r_half[change_point:] *= factor_half

# Apply the change to the data with factor of 2
data_v_double = data_v.copy()
data_y_double = data_y.copy()
data_r_double = data_r.copy()
data_v_double[change_point:] *= factor_double
data_y_double[change_point:] *= factor_double
data_r_double[change_point:] *= factor_double

# Calculate RMSE for each alpha value using test_ewa function
rmse_vimeo_half = test_ewa(data_v_half, alphas)
rmse_youtube_half = test_ewa(data_y_half, alphas)
rmse_reddit_half = test_ewa(data_r_half, alphas)

rmse_vimeo_double = test_ewa(data_v_double, alphas)
rmse_youtube_double = test_ewa(data_y_double, alphas)
rmse_reddit_double = test_ewa(data_r_double, alphas)

# Plot RMSE vs. Alpha for factor of half
plt.figure(figsize=(10, 6))
plt.plot(alphas, rmse_vimeo_half, label='Vimeo')
plt.plot(alphas, rmse_youtube_half, label='YouTube')
plt.plot(alphas, rmse_reddit_half, label='Reddit')
plt.xlabel('Weight of New Sample (Alpha)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('RMSE vs. Alpha for EWA After Halving RTT')
plt.legend()
plt.grid(True)
plt.show()

# Plot RMSE vs. Alpha for factor of 2
plt.figure(figsize=(10, 6))
plt.plot(alphas, rmse_vimeo_double, label='RMSE Vimeo')
plt.plot(alphas, rmse_youtube_double, label='RMSE Youtube')
plt.plot(alphas, rmse_reddit_double, label='RMSE Reddit')
plt.xlabel('Weight of New Sample (Alpha)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('RMSE vs. Alpha After Doubling RTT')
plt.legend()
plt.grid(True)
plt.show()