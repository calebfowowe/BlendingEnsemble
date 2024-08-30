# Data manipulation libraries
import pandas as pd
import numpy as np

# Data visualization Library
import plotly.express as px
import plotly.graph_objects as go
px.height, px.width = 600, 600

# Data cleaning and imputation module
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

# Filepath manager
from pathlib import Path
from datetime import datetime
import sklearn as sk

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
#creates an output directory within the system where charts, and plots are stored.

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Defining random state
import random
def rnd_state(seed=30):
    return seed


#Function to create a folder where outputs of the projects are stored within the local drive
def getpath(name=None):
    if name is not None:
        #Ouput Files paths
        PATH = Path() / "BlendingModelOutput"/ f"{name}"
        PATH.mkdir(parents=True, exist_ok=True)
    else:
        # defined path within path
        PATH = Path() / "BlendingModelOutput"
        PATH.mkdir(parents=True, exist_ok=True)
    return PATH


#Class-weights imbalance
def cwts(dfs):
    c0, c1 = np.bincount(dfs['predict'])
    w0 = (1 / c0) * (len(dfs)) / 2
    w1 = (1 / c1) * (len(dfs)) / 2
    return {0: w0, 1: w1}

def traintestsplit(X, y, test_size=0.2, shuffle=False):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    return Xtrain, Xtest, ytrain, ytest


class LoadData:
    """
    This class is used in loading the data from where it is saved, checking for missing data,
    and method for filling missing data.

    It takes as input:
    dictionary of the filenames as keys, and dateformat as values.
    The first filename and format should be for the stock data

    It also has the option to provide the argument for how nan columns should be filled.
    """
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    @staticmethod
    def adjust_to_friday(date):
        #If date is on a weekend, adjust it to previous business day
        if date.weekday() == 5: #5 = Saturday
            date = date - pd.Timedelta(days=1)
        elif date.weekday() == 6: #6 = Sunday
            date = date - pd.Timedelta(days=2)
        return date

    def getData(self, filename):
        df = pd.read_csv(f'./data/{filename}.csv')
        #Check if the current index is a datetime type
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            match df.columns.any():
                case 'date':
                    df['date'] = pd.to_datetime(df['date'], format='mixed')  # convert to datetime
                    df.set_index('date', inplace=True) #set 'date' column as the index
                case 'dates':
                    df['dates'] = pd.to_datetime(df['dates'], format='mixed')  # convert to datetime
                    df.set_index('dates', inplace=True) #set 'dates' column as the index
                case 'Date':
                    df['Date'] = pd.to_datetime(df['Date'], format='mixed')  # convert to datetime
                    df.set_index('Date', inplace=True) #set 'Date' column as the index
                case 'Dates':
                    df['Dates'] = pd.to_datetime(df['Dates'], format='mixed')  # convert to datetime
                    df.set_index('Dates', inplace=True) #set 'Dates' column as the index
                case 'datetime':
                    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')  # convert to datetime
                    df.set_index('datetime', inplace=True) #set 'datetime' column as the index
                case 'Datetime':
                    df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed')  # convert to datetime
                    df.set_index('Datetime', inplace=True) #set 'Datetime' column as the index
                case _default:
                    raise KeyError("No datetime index or 'date' column found in the DataFrame")
        df.index = pd.to_datetime(df.index, format='%Y-%d-%m')
        df.sort_index(ascending=True, inplace=True)
        df.index = df.index.map(self.adjust_to_friday) #Adjust non_weekday dates to end of month
        return df

    def knn_impute_nan(self, df, n_neighbours=5, weights='uniform'):
        imputer = KNNImputer(n_neighbors=n_neighbours, weights=weights)
        df_imputed = imputer.fit_transform(df)
        return df_imputed


    def joinData(self):
        global add_file, fname
        file_dict = self.kwargs  #unpack the keyword arguments

        #Stock data information must be the first in the frame and should be extracted to the dataframe first
        df_merged = self.getData(file_dict['files'][0])  #Extract historical stock-price data from file list

        # Check if additional data is provided and iterate through to get the data and merge to the stock data
        if len(file_dict['files']) > 1:
            # iterate through other the values and merge to dataframe
            for key, value_list in file_dict.items():
                for fname in value_list[1:]:
                    add_file = self.getData(fname)
                    df_merged = df_merged.join(add_file, how='left', rsuffix=f'_{fname}')

        # Select the time_range specified to be used for the analysis from the finally merged dataframe
        df_merged = df_merged[self.args[0]: self.args[1]]
        #Remove duplicated dates.
        df_merged = df_merged[~df_merged.index.duplicated(keep='first')] #remove duplicated dates
        self.df_new = df_merged.copy()
        return self.df_new

    @staticmethod
    def checkNullData(df):
        rep = df[df.isnull().any(axis=1)]
        return rep

    @staticmethod
    def nanDataCol(df):
        col_with_nan = df.columns[df.isna().any()].tolist()
        return col_with_nan

    def fixNullData(self, data, method=None):
        if method is None:
            method = self.args[2]
        else:
            method = method
        df = data.copy()
        nan_columns = self.nanDataCol(df)
        match method:
            case 'median':
                [df.fillna({col: df[col].median()}, inplace=True) for col in nan_columns]
            case 'mean':
                [df.fillna({col: df[col].mean()}, inplace=True) for col in nan_columns]
            case 'std':
                [df.fillna({col: df[col].std()}, inplace=True) for col in nan_columns]
            case 'bfill':
                [df.fillna({col: df[col].bfill()}, inplace=True) for col in nan_columns]
            case 'ffill':
                [df.fillna({col: df[col].ffill()}, inplace=True) for col in nan_columns]
            case 'droprows':
                df.dropna(axis=0, how='any', inplace=True)
            case 'dropcols':
                df.dropna(axis=1, how='any', inplace=True)
            case 'knnimpute':
                # Refine data and impute missing values
                cols = df.columns.tolist() #Extract column names
                # run the KNNimputer function with 5 neighbours and uniform weights
                df_impute = df.copy()
                df_impute = self.knn_impute_nan(df_impute)

                # convert to numpy output of the transformed data to dataframe
                df_new = pd.DataFrame(df_impute, index=df.index, columns=cols)
                df = df_new.copy()
            case _default:
                raise NotImplementedError(f'wrong variable')
        return df


    def plotCandleStick(self, data, events=None):
        df = data.copy()
        fig = go.Figure(data=[go.Ohlc(x=df.index,
                                      open=df['Open'],
                                      high=df['High'],
                                      low=df['Low'],
                                      close=df['Close'])])

        if events is not None:
            event_dates = events['event_dates']
            event_title = events['event_title']
            #stock_split_dates = ['2024-06-07', '2007-09-11', '2006-04-07', '2001-09-12', '2000-06-27']

            for date in event_dates:
                fig.add_shape(type='line',
                              x0=date, x1=date,
                              y0=0, y1 = 1,
                              xref='x', yref='paper',
                              line=dict(color='black', width=2, dash='dash'))

                # Add annotation
                fig.add_annotation(x=date, y=0.5, xref='x', yref='paper',
                                   text=f"{event_title}", showarrow=False,
                                   textangle=90,
                                   xanchor='left', yanchor='top',
                                   font=dict(color="purple", size=12))

        fig.update_layout(title=f'Candlestick of Price data Open, High, Low and Close',
                          xaxis={'title':'Date', 'color': 'black', 'zeroline': True},
                          yaxis={'title':'Prices ($)', 'color': 'black', 'zeroline': True},
                          xaxis_rangeslider_visible=False,
                          width=900, height=500, plot_bgcolor='rgba(255, 255, 255, 1)',
                        paper_bgcolor='rgba(255, 255, 255, 1)')
        fig.write_html(f"{getpath()}/Candle stick of {self.kwargs['files'][0]}_{datetime.now()}.html")
        return fig.show()

    def plotPrices(self, data):
        stock_name = self.kwargs['files'][0]
        df = data.copy()
        fig = go.Figure()

        # Add traces for each column
        fig.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='lines', name='Open'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['High'], mode='lines', name='High'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Low'], mode='lines', name='Low'))

        fig.update_layout(
            updatemenus=[
                {
                    'buttons': [
                        {
                            'args': [{'y': [df['Open']]}],
                            'label': 'Open',
                            'method': 'restyle'
                        },
                        {
                            'args': [{'y': [df['Close']]}],
                            'label': 'Close',
                            'method': 'restyle'
                        },
                        {
                            'args': [{'y': [df['High']]}],
                            'label': 'High',
                            'method': 'restyle'
                        },
                        {
                            'args': [{'y': [df['Low']]}],
                            'label': 'Low',
                            'method': 'restyle'
                        }
                    ],
                    'direction': 'down',
                    'showactive': True,
                }
            ]
        )
        # Set the initial visible trace
        fig.update_traces(visible=False)
        fig.update_traces(visible=True, selector=dict(name='Close'))

        # Update layout with title and axis labels
        fig.update_layout(
            title=f"{stock_name} Stock Price Data",
            xaxis={'title':'Date', 'color': 'black', 'zeroline': True},
            yaxis={'title':f'Value', 'color': 'black', 'zeroline': True},
            width=900, height=500,
            showlegend=False, plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)')
        fig.write_html(f"{getpath()}/Prices plots_{datetime.now()}.html")
        return fig.show()


