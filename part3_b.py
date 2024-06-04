import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def simulate_rtt_change(data, alpha, change_point, factor):
    srtt = data[0]
    predictions = [srtt]
    for i, rtt in enumerate(data[1:]):
        if i == change_point:
            rtt *= factor
        srtt = (1 - alpha) * srtt + alpha * rtt
        predictions.append(srtt)
    return predictions

# Define change point and factor
change_point = len(data_v) // 2
factor = 2  # For doubling

# Calculate predictions for different alphas
convergence_data = {alpha: simulate_rtt_change(data_v, alpha, change_point, factor) for alpha in alphas}

# Plot convergence
plt.figure(figsize=(10, 6))
for alpha, predictions in convergence_data.items():
    plt.plot(predictions, label=f'Alpha {alpha:.1f}')
plt.axvline(change_point, color='red', linestyle='--', label='Change Point')
plt.xlabel('Sample Index')
plt.ylabel('RTT (seconds)')
plt.title('EWA Convergence After Doubling RTT (Vimeo)')
plt.legend()
plt.grid(True)
plt.show()


factor = 0.5  # For halving

# Calculate predictions for different alphas
convergence_data_half = {alpha: simulate_rtt_change(data_v, alpha, change_point, factor) for alpha in alphas}

# Plot convergence
plt.figure(figsize=(10, 6))
for alpha, predictions in convergence_data_half.items():
    plt.plot(predictions, label=f'Alpha {alpha:.1f}')
plt.axvline(change_point, color='red', linestyle='--', label='Change Point')
plt.xlabel('Sample Index')
plt.ylabel('RTT (seconds)')
plt.title('EWA Convergence After Halving RTT (Vimeo)')
plt.legend()
plt.grid(True)
plt.show()
