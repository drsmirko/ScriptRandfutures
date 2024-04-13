# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:19:58 2024

@author: deros
"""

import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set the ticker symbol for the future to be analyzed
ticker_symbol='6Z=F'
data = yf.download(ticker_symbol)

# Define the contract quantity
n = 500000

# Initialize new DataFrames
cln_data = pd.DataFrame()
results1 = pd.DataFrame()
results = pd.DataFrame()

# Drop any rows with missing data
cln_data = data.dropna()

# Calculate the Profit & Loss
PL = (cln_data['Close'] - cln_data['Close'].shift(1)) * n
cln_data['P&L'] = PL
results = cln_data.dropna()
print(results)

# Calculate the standard deviation for P&L data
dev_std = np.std(results['P&L'][1:], ddof=1)
print("The standard deviation, for a long position is:", dev_std)

# Calculate the standard deviation over two days, using square root of time scaling
dev_std_2 = (dev_std * np.sqrt(2))
print("The standard deviation over two days, for a long position is:", dev_std_2)

# Set the value for cumulative probability
prob = 0.99

# Calculate the z-value for the given probability
N_x = norm.ppf(prob)
print("The x value corresponding to the cumulative probability of", prob, "is:", N_x)

# Calculate the standard deviation multiplied by the z-value
Nx_Std1 = dev_std * N_x
Nx_Std2 = dev_std_2 * N_x

print('The standardized normal deviation for one day is', Nx_Std1)
print('The standardized normal deviation for two days is', Nx_Std2)

# Calculate and round the initial margin required
Dpg_1 = round(Nx_Std1)
Dpg_2 = round(Nx_Std2)

print('The Initial Margin for one day is', Dpg_1)
print('The Initial Margin calculated for two days is', Dpg_2)

# Calculate and round the maintenance margin
Mrgi_1 = (Dpg_1 * 0.75)
Mrgi_2 = (Dpg_2 * 0.75)
Mrgia_1 = round(Mrgi_1)
Mrgia_2 = round(Mrgi_2)

print('The Maintenance Margin calculated for one day is', Mrgia_1)
print('The Maintenance Margin calculated for two days is', Mrgia_2)

# Count the occurrences where the P&L plus one-day margin is less than the maintenance margin
conteggio_1 = (Dpg_1 + PL < Mrgia_1).sum()

# Count the occurrences where the P&L plus two-day margin is less than the maintenance margin
conteggio_2 = (Dpg_2 + PL < Mrgia_2).sum()

print(f"The number of times the one-day Initial Margin plus P&L is less than the maintenance margin is: {conteggio_1}")
print(f"The number of times the two-day Initial Margin plus P&L is less than the maintenance margin is: {conteggio_2}")

# Count the occurrences where the P&L plus one-day margin is less than zero
WA_1 = (Dpg_1 + PL < 0).sum()
print(f"The number of times the one-day Initial Margin plus P&L is less than 0 is: {WA_1}")

# Count the occurrences where the P&L plus two-day margin is less than zero
WA_2 = (Dpg_2 + PL < 0).sum()
print(f"The number of times the two-day Initial Margin plus P&L is less than 0 is: {WA_2}")

# Evaluate the percentage of occasions when the initial margin is completely depleted
P_1 = (WA_1 / len(results['P&L'])) * 100
P_2 = (WA_2 / len(results['P&L'])) * 100
P_3 = (conteggio_1 / len(results['P&L'])) * 100
P_4 = (conteggio_2 / len(results['P&L'])) * 100

print(f"The percentage for a walk away with a one-day initial margin is: {P_1}%")
print(f"The percentage for a walk away with a two-day initial margin is: {P_2}%")
print(f"The percentage for of one-day Initial Margin plus P&L is less than the maintenance margin is: {P_3}%")
print(f"The percentage for of two-day Initial Margin plus P&L is less than the maintenance margin is: {P_4}%")

# Set the training set to 60% of the data
T_set= int(len(results) * 0.6)

# Select the first 60% of the rows as the training set
subset_data = cln_data.iloc[:T_set]
print(subset_data)

# Calculate logarithmic returns
subset_data['Returns'] = np.log(subset_data['Close'] / subset_data['Close'].shift(1))

# Replace infinite values with NaN and keep non-NaN values
subset_data['Returns'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Clean data by removing rows with NaN values
subset_data.dropna(subset='Returns', inplace=True)

# Check data after cleaning
if subset_data['Returns'].isnull().values.any() or np.isinf(subset_data['Returns']).values.any():
    raise ValueError("The returns data contains NaN or infinite values.")

# Lambda parameter for the EWMA model
lambda_ = 0.94

# Define the EWMA volatility calculation function
def calculate_ewma_volatility(returns, lambda_):
    ewma_volatility = np.zeros_like(returns)
    ewma_volatility[0] = np.var(returns)
    for t in range(1, len(returns)):
        ewma_volatility[t] = lambda_ * ewma_volatility[t-1] + (1 - lambda_) * returns[t-1]**2
    return ewma_volatility

# Define the function to calculate log-likelihood for the EWMA model
def ewma_log_likelihood(returns, ewma_volatility):
    m = len(returns)
    variance = ewma_volatility ** 2
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance) + (returns ** 2) / variance)
    return log_likelihood

# Calculate EWMA volatility
subset_data['EWMA_Volatility'] = calculate_ewma_volatility(subset_data['Returns'].values, lambda_)

# Check EWMA volatility for NaN or infinite values
if subset_data['EWMA_Volatility'].isnull().values.any() or np.isinf(subset_data['EWMA_Volatility']).values.any():
    raise ValueError("The EWMA volatility data contains NaN or infinite values.")

# Fit the GARCH(1,1) model and calculate log-likelihood for GARCH and EWMA models
model = arch_model(subset_data['Returns'], p=1, q=1, mean='zero')
model_fitted = model.fit(disp='off')
garch_log_likelihood = model_fitted.loglikelihood

# Calculate log-likelihood for the EWMA model
ewma_volatility = subset_data['EWMA_Volatility'].values[1:] 
returns = subset_data['Returns'].values[1:]  
ewma_ll = ewma_log_likelihood(returns, ewma_volatility)

# Compare based on log-likelihood
if ewma_ll > garch_log_likelihood:
    print("EWMA model has a higher log-likelihood and is preferred.")
else:
    print("GARCH model has a higher log-likelihood and is preferred.")

# Clean up results DataFrame
results.dropna(axis=1, how='any', inplace=True)

# Define the validation set
subset_data_2 = cln_data.iloc[T_set:]

# Fit the GARCH(1,1) model to the validation set and calculate historical and GARCH volatility
model_3 = arch_model(subset_data_2['P&L'], p=1, q=1, mean='zero')
model_fitted_3 = model_3.fit(disp='off')
subset_data_2['GARCH_Volatility'] = model_fitted_3.conditional_volatility
subset_data_2['Historical_Volatility'] = np.std(subset_data_2['P&L'], ddof=1)

# Ensure to drop any NaNs created from the volatility calculations
subset_data_2.dropna(subset=['Historical_Volatility', 'GARCH_Volatility'], inplace=True)

# Calculate error: subtract the predicted volatility from the historical and square it
errors = (subset_data_2['Historical_Volatility'] - subset_data_2['GARCH_Volatility']) ** 2

# Calculate MSE
mse = errors.mean()
print(f"MSE: {mse}")

# Calculate RMSE
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

# Calculate historical margins using a rolling window of 20 days
window_size = 20
results['Historical_Margin'] = np.nan
for i in range(0, len(results) - window_size + 1, window_size):
    start_date = results.index[i]
    end_date = results.index[min(i + window_size - 1, len(results) - 1)]
    window = results.loc[start_date:end_date]
    dev_std_window = np.std(window['P&L'])
    Nx_Std_window = dev_std_window * norm.ppf(prob)
    Dpg_window = round(Nx_Std_window)
    Mrgi_window = round(Dpg_window * 0.75)
    results.loc[start_date:end_date, 'Historical_Margin'] = Mrgi_window

# Fill missing values at the end if necessary
if np.isnan(results['Historical_Margin'].iloc[-1]):
    results['Historical_Margin'].fillna(method='ffill', inplace=True)

# Fit the GARCH(1,1) model to the entire dataset
model_2 = arch_model(results['P&L'], p=1, q=1, mean='zero')
model_fitted_2 = model_2.fit(disp='off')

# Calculate the initial margin based on GARCH volatility
results['GARCH_Volatility'] = model_fitted_2.conditional_volatility
print(results['GARCH_Volatility'])
results['GARCH_Margin'] = np.nan
for i in range(len(results)):
    Nx_Std_garch = results['GARCH_Volatility'].iloc[i] * norm.ppf(prob)
    Dpg_garch = round(Nx_Std_garch)
    Mrgi_garch = round(Dpg_garch * 0.75)
    results.loc[results.index[i], 'GARCH_Margin'] = Mrgi_garch
   
# Compare the two sets of margins
print(results[['Historical_Margin', 'GARCH_Margin']].tail())
print(results['Historical_Margin'])

# Initialize the 'Historical_Margin' column with NaN
results['Monthly_Margin'] = np.nan

# Calculate the margin for each month
for name, group in results.groupby(pd.Grouper(freq='M')):
    dev_std_month = np.std(group['P&L'])  # Standard deviation of P&L for the month
    Nx_Std_month = dev_std_month * norm.ppf(prob)  # Calculate the modified normal standard deviation
    Dpg_month = round(Nx_Std_month)  # Round the result
    Mrg_month = round(Dpg_month * 0.75)  # Calculate 75% of the rounded value
    results.loc[group.index, 'Monthly_Margin'] = Mrg_month  # Assign the calculated margin to the DataFrame

# Create plots
plt.figure(figsize=(10, 6))
plt.plot(results.index, results['Historical_Margin'], label='Historical Maintenance Margin', color='blue')
plt.title('Maintenance Margin Over Time')
plt.xlabel('Date')
plt.ylabel('Maintenance Margin $')
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate() 
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(results.index, results['Historical_Margin'], label='Historical Margin', color='blue')
plt.plot(results.index, results['GARCH_Margin'], label='GARCH Margin', color='green')
plt.title('Comparison of Historical vs GARCH Maintenance Margins')
plt.xlabel('Date')
plt.ylabel('Maintenance Margin ($)')
plt.legend()
plt.grid(True)
plt.show()