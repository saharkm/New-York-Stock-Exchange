#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import SeasonalAD
from adtk.detector import ThresholdAD


from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning,ModelWarning
warnings.simplefilter('ignore', ValueWarning)
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', ModelWarning)



def create_date_index (data):
    """
    This Function will convert the date column to index with DatetimeIndex
    """
    data.index = pd.DatetimeIndex(data['date'])
    data.drop(['symbol','date'], axis = 1, inplace = True)
    print(f'The Index of dataframe is now date with the type of {type(data.index)}')
    return data

def fillup_days (data):
    """
    This Function will add the non-trading days to dataframe index and fill up the paramteres
    for that days from last available date values
    """
    min_date = min(data.index)
    max_date = max(data.index)
    total_days = max_date - min_date
    non_trading_days  = total_days.days - len(data)
    print(f'There are {non_trading_days} days without data. These days will be filled up using forward filling method...')
    idx = pd.date_range(min_date, max_date)
    data = data.reindex(idx)
    data.fillna(method="ffill", inplace = True)
    data = validate_series(data)
    print(f'Non-trading days filled up. The dataframe has now {len(data)} records')
    return data

def seasonal_anomaly_detector(data, parameter):
    """
    This function will find the seasonal anomaly using ADTK library for a specific parameter
    """
    seasonal = SeasonalAD()
    try:
        col_name = ['open', 'close', 'low', 'high', 'volume', 'close_change']
        anomalies = seasonal.fit_detect(data[parameter])
        print(f'{sum(anomalies)} records are detected as {parameter} seasonal anomaly.')
        new_parameter = parameter + '_seasonal_anomaly'
        data[new_parameter] = anomalies
        print(f'{new_parameter} column was added to dataframe.')
        plot(data[col_name], anomaly=anomalies, anomaly_color="orange", anomaly_tag="marker")
        return data
    except Exception:
        print(f'Could not find significant seasonality. No column was added.')
        return data

def threshhold_anomaly_detector (data, parameter, sd_threshhold):
    """
    This function will find the threshhold anomaly using ADTK library for a specific parameter
    and given standard deviation reference
    """
    col_name = ['open', 'close', 'low', 'high', 'volume', 'close_change']
    threshold_high = np.mean(data[parameter]) + sd_threshhold * np.std(data[parameter])
    print(f'High threshhold for {parameter} is set to {np.round(threshold_high,decimals= 2)}')
    threshold_low = np.mean(data[parameter]) - sd_threshhold * np.std(data[parameter])
    print(f'Low threshhold for {parameter} is set to {np.round(threshold_low, decimals=2)}')
    threshold_val = ThresholdAD(high=threshold_high, low=threshold_low)
    anomalies = threshold_val.detect(data[parameter])
    print(f'{sum(anomalies)} records are detected as {parameter} threshhold anomaly.')
    new_parameter = parameter + '_threshhold_anomaly'
    data[new_parameter] = anomalies
    print(f'{new_parameter} column was added to dataframe.')
    plot(data[col_name], anomaly=anomalies, anomaly_color="orange", anomaly_tag="marker")
    return data

def train_test_Split(series, ratio):
    """
    This Function will split the series to train and test based on ratio
    """
    series_len = len(series)
    train_len = int(series_len * ratio)
    train_series = series[:train_len]
    test_series = series[train_len:]
    return (train_series, test_series)

def normalize_min_max ( series):
    """
    This Function will return a normalized series based on min & max method.
    """
    normal_series = (series - series.min()) / (series.max() - series.min())
    return normal_series

def sarimax_model (train_series, test_series, order ,seasonal_order, trend ):
    ARMAmodel = SARIMAX(train_series, order=order,seasonal_order=seasonal_order, trend = trend)
    ARMAmodel = ARMAmodel.fit()
    y_pred = ARMAmodel.get_forecast(len(test_series.index))
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df.index = test_series.index
    y_pred_out = y_pred_df["Predictions"]
    rms = mean_squared_error(test_series.values, y_pred_out, squared=False)
    print(f"Mean square root error is: {rms}" )
    return (ARMAmodel, rms, y_pred_out)

def best_model_order (train_series, test_series, order_1: list, order_2: list, order_3: list, seasonal_order):
    lst_order_1= []
    lst_order_2= []
    lst_order_3= []
    lst_rms = []
    for i in order_1:
        for j in order_2:
            for t in order_3:
                result = sarimax_model(train_series, test_series, order = (i, j, t), seasonal_order=seasonal_order, trend = 'ct')
                lst_order_1.append(i)
                lst_order_2.append(j)
                lst_order_3.append(t)
                lst_rms.append(result[1])
    best_index = lst_rms.index(min(lst_rms))
    print(f'Best model rms is: {lst_rms[best_index]}.\n The parameters are:\norder #1:  {lst_order_1[best_index]}\norder #2:  {lst_order_2[best_index]}\norder #3:  {lst_order_3[best_index]}\n')
    return {
        'rms':lst_rms[best_index] ,
        'order': (lst_order_1[best_index],lst_order_2[best_index], lst_order_3[best_index])
    }


def best_model (train_series, test_series, order: tuple, seas_1: list, seas_2: list, seas_3: list, seas_4: list):
    lst_seas_1= []
    lst_seas_2= []
    lst_seas_3= []
    lst_seas_4= []
    lst_rms = []
    for i in seas_1:
        for j in seas_2:
            for k in seas_3:
                for t in seas_4:
                    result = sarimax_model(train_series, test_series, order = order, seasonal_order=(i, j, k, t), trend = 'ct')
                    lst_seas_1.append(i)
                    lst_seas_2.append(j)
                    lst_seas_3.append(k)
                    lst_seas_4.append(t)
                    lst_rms.append(result[1])
    best_index = lst_rms.index(min(lst_rms))
    print(
        f'Best model rms is:  {lst_rms[best_index]}\n The parameters are:\nSeasonality order #1:  {lst_seas_1[best_index]}\nSeasonality order #2:  {lst_seas_2[best_index]}\nSeasonality order #3:  {lst_seas_3[best_index]}\nSeasonality order #4:  {lst_seas_4[best_index]}\n'
    )
    return {
        'rms':lst_rms[best_index] ,
        'seasonality': (lst_seas_1[best_index],lst_seas_2[best_index], lst_seas_3[best_index], lst_seas_4[best_index])
    }