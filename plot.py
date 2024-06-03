import pandas as pd
import matplotlib.pyplot as plt

data_v = pd.read_csv('Vimeo.csv')
data_y = pd.read_csv('Youtube.csv')
data_r = pd.read_csv('Reddit.csv')

# Find the length of the shortest dataframe
min_length = min(len(data_v), len(data_y), len(data_r))

# Trim the dataframes to the length of the shortest dataframe
data_v = data_v[:min_length]
data_y = data_y[:min_length]
data_r = data_r[:min_length]

plt.plot(data_v['RTT'], label='Vimeo')
plt.plot(data_y['RTT'], label='Youtube')
plt.plot(data_r['RTT'], label='Reddit')
plt.ylabel('RTT (seconds)')
plt.title('TCP RTT over Time')
plt.grid(True)
plt.legend()
plt.show()