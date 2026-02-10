import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_treatment import clean_inmet_2025_df
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = clean_inmet_2025_df()

# Obligatory to remove NaNs/empty/invalid values to avoid errors in the regression
# An analysis of the data was already done and I know it's safe to drop the NaN here.
# No need to substitute with averages or do any additional investigation
df.dropna(inplace=True)

# Separate dependent/target (Y) and independent (X) variables
# Remember to remove any redundant variables from the regression analyses.
X = df.loc[:,['Dir. Vento (°)','Temp. Ins. (C)']].to_numpy()
y = df['Vel. Vento (m/s)'].to_numpy()

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
