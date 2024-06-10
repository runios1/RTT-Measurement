from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

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
X_v = np.arange(len(data_v)).reshape(-1, 1)
y_v = data_v

X_y = np.arange(len(data_y)).reshape(-1, 1)
y_y = data_y

X_r = np.arange(len(data_r)).reshape(-1, 1)
y_r = data_r

# Train the model
model_v = LinearRegression().fit(X_v, y_v)

model_y = LinearRegression().fit(X_y, y_y)

model_r = LinearRegression().fit(X_r, y_r)

# Predict RTT
predictions_lr_v = model_v.predict(X_v)

predictions_lr_y = model_y.predict(X_y)

predictions_lr_r = model_r.predict(X_r)

# Calculate RMSE
rmse_lr_v = calculate_rmse(y_v, predictions_lr_v)

rmse_lr_y = calculate_rmse(y_y, predictions_lr_y)

rmse_lr_r = calculate_rmse(y_r, predictions_lr_r)

# Polynomial Regression
poly = PolynomialFeatures(degree=100)

X_poly_v = poly.fit_transform(X_v)
X_poly_y = poly.fit_transform(X_y)
X_poly_r = poly.fit_transform(X_r)

model_poly_v = LinearRegression().fit(X_poly_v, y_v)
model_poly_y = LinearRegression().fit(X_poly_y, y_y)
model_poly_r = LinearRegression().fit(X_poly_r, y_r)

predictions_poly_v = model_poly_v.predict(X_poly_v)
predictions_poly_y = model_poly_y.predict(X_poly_y)
predictions_poly_r = model_poly_r.predict(X_poly_r)

rmse_poly_v = calculate_rmse(y_v, predictions_poly_v)
rmse_poly_y = calculate_rmse(y_y, predictions_poly_y)
rmse_poly_r = calculate_rmse(y_r, predictions_poly_r)

# Plot predictions
# plt.figure(figsize=(10, 6))
# plt.plot(y, label='Actual RTT')
# plt.plot(predictions_lr, label='Linear Regression Prediction')
# plt.xlabel('Sample Index')
# plt.ylabel('RTT (seconds)')
# plt.title('RTT Prediction Using Linear Regression (Vimeo)')
# plt.legend()
# plt.grid(True)
# plt.show()

print(f'RMSE for Linear Regression in Vimeo: {rmse_lr_v:.4f}')
print(f'RMSE for Linear Regression in Youtube: {rmse_lr_y:.4f}')
print(f'RMSE for Linear Regression in Reddit: {rmse_lr_r:.4f}')

print(f'RMSE for Polynomial Regression in Vimeo: {rmse_poly_v:.4f}')
print(f'RMSE for Polynomial Regression in Youtube: {rmse_poly_y:.4f}')
print(f'RMSE for Polynomial Regression in Reddit: {rmse_poly_r:.4f}')


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

# Prepare data for linear regression
X_v_half = np.arange(len(data_v_half)).reshape(-1, 1)
y_v_half = data_v_half

X_y_half = np.arange(len(data_y_half)).reshape(-1, 1)
y_y_half = data_y_half

X_r_half = np.arange(len(data_r_half)).reshape(-1, 1)
y_r_half = data_r_half

X_v_double = np.arange(len(data_v_double)).reshape(-1, 1)
y_v_double = data_v_double

X_y_double = np.arange(len(data_y_double)).reshape(-1, 1)
y_y_double = data_y_double

X_r_double = np.arange(len(data_r_double)).reshape(-1, 1)
y_r_double = data_r_double

# Train the model
model_v_half = LinearRegression().fit(X_v_half, y_v_half)
model_y_half = LinearRegression().fit(X_y_half, y_y_half)
model_r_half = LinearRegression().fit(X_r_half, y_r_half)

model_v_double = LinearRegression().fit(X_v_double, y_v_double)
model_y_double = LinearRegression().fit(X_y_double, y_y_double)
model_r_double = LinearRegression().fit(X_r_double, y_r_double)

# Predict RTT
predictions_lr_v_half = model_v_half.predict(X_v_half)
predictions_lr_y_half = model_y_half.predict(X_y_half)
predictions_lr_r_half = model_r_half.predict(X_r_half)

predictions_lr_v_double = model_v_double.predict(X_v_double)
predictions_lr_y_double = model_y_double.predict(X_y_double)
predictions_lr_r_double = model_r_double.predict(X_r_double)

# Calculate RMSE
rmse_lr_v_half = calculate_rmse(y_v_half, predictions_lr_v_half)
rmse_lr_y_half = calculate_rmse(y_y_half, predictions_lr_y_half)
rmse_lr_r_half = calculate_rmse(y_r_half, predictions_lr_r_half)

rmse_lr_v_double = calculate_rmse(y_v_double, predictions_lr_v_double)
rmse_lr_y_double = calculate_rmse(y_y_double, predictions_lr_y_double)
rmse_lr_r_double = calculate_rmse(y_r_double, predictions_lr_r_double)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)

X_poly_v_half = poly.fit_transform(X_v_half)
X_poly_y_half = poly.fit_transform(X_y_half)
X_poly_r_half = poly.fit_transform(X_r_half)

model_poly_v_half = LinearRegression().fit(X_poly_v_half, y_v_half)
model_poly_y_half = LinearRegression().fit(X_poly_y_half, y_y_half)
model_poly_r_half = LinearRegression().fit(X_poly_r_half, y_r_half)

X_poly_v_double = poly.fit_transform(X_v_double)
X_poly_y_double = poly.fit_transform(X_y_double)
X_poly_r_double = poly.fit_transform(X_r_double)

model_poly_v_double = LinearRegression().fit(X_poly_v_double, y_v_double)
model_poly_y_double = LinearRegression().fit(X_poly_y_double, y_y_double)
model_poly_r_double = LinearRegression().fit(X_poly_r_double, y_r_double)

# Predict RTT
predictions_poly_v_half = model_poly_v_half.predict(X_poly_v_half)
predictions_poly_y_half = model_poly_y_half.predict(X_poly_y_half)
predictions_poly_r_half = model_poly_r_half.predict(X_poly_r_half)

predictions_poly_v_double = model_poly_v_double.predict(X_poly_v_double)
predictions_poly_y_double = model_poly_y_double.predict(X_poly_y_double)
predictions_poly_r_double = model_poly_r_double.predict(X_poly_r_double)

# Calculate RMSE
rmse_poly_v_half = calculate_rmse(y_v_half, predictions_poly_v_half)
rmse_poly_y_half = calculate_rmse(y_y_half, predictions_poly_y_half)
rmse_poly_r_half = calculate_rmse(y_r_half, predictions_poly_r_half)

rmse_poly_v_double = calculate_rmse(y_v_double, predictions_poly_v_double)
rmse_poly_y_double = calculate_rmse(y_y_double, predictions_poly_y_double)
rmse_poly_r_double = calculate_rmse(y_r_double, predictions_poly_r_double)

# Print RMSE values
print(f'RMSE for Linear Regression in Vimeo (half RTT): {rmse_lr_v_half:.4f}')
print(f'RMSE for Linear Regression in Youtube (half RTT): {rmse_lr_y_half:.4f}')
print(f'RMSE for Linear Regression in Reddit (half RTT): {rmse_lr_r_half:.4f}')

print(f'RMSE for Polynomial Regression in Vimeo (half RTT): {rmse_poly_v_half:.4f}')
print(f'RMSE for Polynomial Regression in Youtube (half RTT): {rmse_poly_y_half:.4f}')
print(f'RMSE for Polynomial Regression in Reddit (half RTT): {rmse_poly_r_half:.4f}')

print(f'RMSE for Linear Regression in Vimeo (double RTT): {rmse_lr_v_double:.4f}')
print(f'RMSE for Linear Regression in Youtube (double RTT): {rmse_lr_y_double:.4f}')
print(f'RMSE for Linear Regression in Reddit (double RTT): {rmse_lr_r_double:.4f}')

print(f'RMSE for Polynomial Regression in Vimeo (double RTT): {rmse_poly_v_double:.4f}')
print(f'RMSE for Polynomial Regression in Youtube (double RTT): {rmse_poly_y_double:.4f}')
print(f'RMSE for Polynomial Regression in Reddit (double RTT): {rmse_poly_r_double:.4f}')