import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_treatment import clean_inmet_astation_data
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

source_file_path_1 = r'A318_2025.csv'
source_file_path_2 = r'A339_2025.csv'
df_A318 = clean_inmet_astation_data(source_file_path_1)
df_A339 = clean_inmet_astation_data(source_file_path_2)

# Obligatory to remove NaNs/empty/invalid values to avoid errors in the regression
# An analysis of the data was already done and I know it's safe to drop the NaN here.
# No need to substitute with averages or do any additional investigation
df_A318.dropna(inplace=True)
df_A339.dropna(inplace=True)

# Separate dependent/target (Y) and independent (X) variables
# Remember to remove any redundant variables from the regression analyses.
X = df_A318.loc[:,['Dir. Vento (°)', 'Temp. Ins. (C)']].to_numpy()
y = df_A318['Vel. Vento (m/s)'].to_numpy()

# Standardize input features so the model doesn't inadvertently favor any feature due to its magnitude.
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)

# Separate test and training data
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=42)

# Create the regression
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Evaluate results

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model evaluation:")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

df_results = pd.DataFrame({
    "y_real": y_test,
    "y_predicted": y_pred,
    "error": y_test - y_pred
})

print("\nData sample:")
print(df_results.head(10))

# Plots
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

plt.xlabel("y")
plt.ylabel("y-hat")
plt.title("Multiple linear regression: Test data (y) vs Predicted value (y-hat)")
plt.grid(True)
plt.show()

residue = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residue, alpha=0.6)
plt.axhline(0, linestyle='--')

plt.xlabel("Predicted value")
plt.ylabel("Residue")
plt.title("Residue analysis")
plt.grid(True)
plt.show()

# To perform correlations between two different datasets, they need to be the same size (number of rows).
# In this case, I will adopt the strategy of aligning both databases according to the time stamp.
# I.e., only data points in time/date that exists in both databases will remain.

df = pd.merge(
    df_A318[["Datetime", "Vel. Vento (m/s)"]],
    df_A339[["Datetime", "Vel. Vento (m/s)"]],
    on="Datetime",
    how="inner"
)

df.columns = ["timestamp", "wind_A318", "wind_A339"]
plt.scatter(df['wind_A339'], df['wind_A318'])
plt.xlabel("Wind speed A339 (m/s)")
plt.ylabel("Wind speed A318 (m/s)")
plt.title("Scatter analysis")
plt.grid(True)
plt.show()

### Start of LR between neighboring met masts
### -----------------------------------------

aX = df['wind_A339'].to_numpy()
ay = df['wind_A318'].to_numpy()

# Standard scaler/fit is not necessary here because both variables have comparable dimensions
# Separate test and training data
aX_train, aX_test, ay_train, ay_test = train_test_split(aX, ay,test_size=0.3,random_state=42)

# Create the regression
aregressor = linear_model.LinearRegression()
aregressor.fit(aX_train.reshape(-1, 1), ay_train)
# Evaluate results
ay_pred = aregressor.predict(aX_test.reshape(-1, 1))
amse = mean_squared_error(ay_test, ay_pred)
armse = np.sqrt(amse)
amae = mean_absolute_error(ay_test, ay_pred)
ar2 = r2_score(ay_test, ay_pred)

print("Model evaluation:")
print(f"MSE  : {amse:.4f}")
print(f"RMSE : {armse:.4f}")
print(f"MAE  : {amae:.4f}")
print(f"R²   : {ar2:.4f}")

adf_results = pd.DataFrame({
    "y_real": ay_test,
    "y_predicted": ay_pred,
    "error": ay_test - ay_pred
})

print("\nData sample:")
print(adf_results.head(10))

# Plots
plt.figure(figsize=(8,6))
plt.scatter(ay_test, ay_pred, alpha=0.6)
plt.plot(
    [ay_test.min(), ay_test.max()],
    [ay_test.min(), ay_test.max()],
    linestyle='--'
)

plt.xlabel("y")
plt.ylabel("y-hat")
plt.title("Multiple linear regression: Test data (y) vs Predicted value (y-hat)")
plt.grid(True)
plt.show()

aresidue = ay_test - ay_pred

plt.figure(figsize=(8,6))
plt.scatter(ay_pred, aresidue, alpha=0.6)
plt.axhline(0, linestyle='--')

plt.xlabel("Predicted value")
plt.ylabel("Residue")
plt.title("Residue analysis")
plt.grid(True)
plt.show()



