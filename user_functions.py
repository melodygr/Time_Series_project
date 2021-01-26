import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import itertools
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib.pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm


def visualize_time_series(df, name):
    """Plot time series. Plot annual breakouts for years with all 12 values."""
    df.plot(figsize = (12,4))
    plt.title(name)
    plt.xlabel('Year')
    plt.ylabel('Median House Price')
    plt.savefig('images/zip_lineplot.png');
    
    # Use pandas grouper to group values using annual frequency
    year_groups = df.groupby(pd.Grouper(freq ='A'))

    # Create a new DataFrame and store yearly values in columns 
    df_annual = pd.DataFrame()

    # print(list(year_groups))
    for yr, group in year_groups:
        if len(group) == 12: # Can only use full years of data
            df_annual[yr.year] = group.values.ravel()
    
    # Plot the yearly groups as subplots
    df_annual.plot(figsize = (13,20), subplots=True, legend=True)
    plt.savefig('images/annual_breakout.png');

    # Plot overlapping yearly groups 
    df_annual.plot(figsize = (15,10), subplots=False, legend=True)
    plt.savefig('images/annual_overlap.png');
    
def visualize_all_series(list_of_df, names):
    """Plot a list of time series dataframes together with provided names for legend."""
    df_group = pd.concat(list_of_df, axis=1)
    df_group.columns = names
    df_group.plot(figsize = (12,4), subplots=False, legend=True)
    plt.title('Median House Prices Over Time')
    plt.xlabel('Year')
    plt.ylabel('Median House Price')
    plt.savefig('images/lineplotallzips.png')
    plt.show();

def stationarity_check(TS):
    """Calculate rolling statistics and plot against original time series, perform and output Dickey Fuller test."""
    # Import adfuller
    from statsmodels.tsa.stattools import adfuller
    
    # Calculate rolling statistics
    roll_mean = TS.rolling(window=24, center=False).mean()
    roll_std = TS.rolling(window=24, center=False).std()
        
    # Perform the Dickey Fuller Test
    dftest = adfuller(TS)
    
    # Plot rolling statistics:
    fig = plt.figure(figsize=(12,6))
    plt.plot(TS, color='blue',label='Original')
    plt.plot(roll_mean.dropna(), color='red', label='Rolling Mean')
    plt.plot(roll_std.dropna(), color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig('images/rolling.png')
    plt.show(block=False)
    
    # Print Dickey-Fuller test results
    print('Results of Dickey-Fuller Test: \n')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 
                                             '#Lags Used', 'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    return None

def run_arima_models(name, train, test, order, metrics_df, seasonal_order = (0,0,0,0)):
    """Runs baseline ARIMA model and adds metrics and results to a passed dataframe"""
    
    model_metrics = [name, order, seasonal_order]
    
    tic = time.time()
    model = ARIMA(train, order=order, seasonal_order=seasonal_order, freq='MS')
    results = model.fit()
    traintime = time.time() - tic
    
    model_metrics.append(round(traintime, 4))
    
    # Print out summary information on the fit
    # print(results.summary())
    
    model_metrics.extend([round(results.params[0], 2), round(results.params[1], 4), 
                          round(results.params[2], 4), round(results.params[3], 2)])
    model_metrics.append(round(results.aic, 2))
    
    # Get predictions starting from first test index and calculate confidence intervals
    # toc = time.time()
    # pred = results.get_prediction(start = test.index[0], end = test.index[-1], dynamic=True, full_results=True)
    # pred_conf = pred.conf_int()
    # predtime = time.time() - toc
    # model_metrics.append(predtime)
    
    # Add model metrics to passed metrics df    
    series = pd.Series(model_metrics, index = metrics_df.columns)
    metrics_df = metrics_df.append(series, ignore_index=True)
    
    return metrics_df

def grid_search_arima(train):
    '''Attempt all pdq parameters to find lowest AIC value'''
    
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 3)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    
    # Generate all different combinations of seasonal p, q and q triplets
    pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    # Run a grid with pdq and seasonal pdq parameters calculated above and get the best AIC value
    ans = []
    for comb in pdq:
        for combs in pdqs:
            try:
                grid_model = ARIMA(train, order=comb, seasonal_order=combs, freq='MS')
                grid_results = grid_model.fit()
                ans.append([comb, combs, grid_results.aic])
    #             print('ARIMA {} x {}12 : AIC Calculated ={}'.format(comb, combs, results.aic))
            except:
                continue
    ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
    print(ans_df.loc[ans_df['aic'].idxmin()])
    
    return ans_df

