# Data manipulation libraries
import traceback
import pandas as pd
import numpy as np

from tabulate import tabulate
from datetime import datetime

# Data visualization Library
import plotly.express as px
import plotly.graph_objects as go

px.height, px.width = 400, 600

#Boruta
from boruta import BorutaPy

#File processor
import sys
from pathlib import Path

from src.utils_data_processing import getpath, rnd_state

#Technical indicator
import pandas_ta as ta

#ML modules
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, RocCurveDisplay, ConfusionMatrixDisplay)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# file logger
from loguru import logger
import sys

output = getpath("logs")

#code logs stored here
logger.remove()
logger.add(sys.stdout, format="{time: MMMM D, YYYY - HH:mm:ss} ----- <level> {message} </level>")
logger.add(f'{output}/blendingmodel.log', serialize=False)


class DayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        self.data = pd.DataFrame(
            {
                'WeekDay': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            }
        )
        self.daysnum = np.array(self.data.index + 1)
        return self
    def transform(self, X):  #X is a dataframe
        Xt = X.copy()
        pi = np.pi
        num = Xt.index.weekday + 1

        Xt['dsin'] = np.sin(2 * pi * num / np.max(self.daysnum))
        Xt['dcos'] = np.cos(2 * pi * num / np.max(self.daysnum))
        Xt = Xt.drop(['days'], axis=1)
        return Xt


class FeaturesEngineering:
    """
    Features Engineering class: subdivided into three subclasses: which inherits the FeaturesEngineering class
    - Features Creation/Generation
    - Features Transformation/Scaling
    - Features Selection
    """

    def __init__(self, dataframe: pd.DataFrame, testsize=0.20):
        self.dataframe = dataframe
        self.testsize = testsize

    # Class-weights imbalances
    def cwts(self, dfs):
        c0, c1 = np.bincount(dfs['predict'])
        w0 = (1 / c0) * (len(dfs)) / 2
        w1 = (1 / c1) * (len(dfs)) / 2
        return {0: w0, 1: w1}

    def traintestsplit(self, X, y, test_size=0.2, shuffle=False):
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        return Xtrain, Xtest, ytrain, ytest

    @staticmethod
    def randomforestSelection(X_train, X_test, y_train, y_test, class_weight, scaler=StandardScaler(), max_features=6):
        # define random forest classifier
        rf = RandomForestClassifier(n_jobs=-1, class_weight=class_weight, random_state=rnd_state(), max_features=max_features)
        # scale and fit the model
        rf_pipe = Pipeline([
            ('transformer', scaler),
            ('classifier', rf)
        ])
        # train the model
        rf_pipe.fit(X_train, y_train)

        # predict
        y_pred = rf_pipe.predict(X_test)
        acc_scores = accuracy_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred, average='weighted')
        class_rpt = classification_report(y_test, y_pred)

        return acc_scores, f1score, class_rpt

    def plot_correlation_matrix(self, X, mthd='pearson'):
        corr_matrix = X.corr(method=mthd)
        # create a Plotly figure
        fig = go.Figure(go.Heatmap(
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            z=corr_matrix.values.tolist(),
            colorscale='blugrn', zmin=0, zmax=1
        ))
        fig.update_traces(text=corr_matrix.values.tolist(), hovertemplate=None)

        # customize the plot
        fig.update_layout(
            title='Features Correlation Matrix',
            width=1350,
            height=1100,
            font=dict(size=12),
            margin=dict(l=100, r=100, t=100, b=100), plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)'
        )
        # show the plot
        fig.write_html(f"{getpath()}/Mutlicollinearity, correlation matrix_{datetime.now()}.html")
        return fig.show()

    @staticmethod
    def correlated_features(data, threshold):
        col_corr = set()
        corr_matrix = data.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
        return col_corr


class FeaturesCreation(FeaturesEngineering):
    """
    Features creation class which takes a stock data in dataframe format, consisting of Open, High, Low,
    Close, and Volume, and returns a host of technical indicators which is built on the Pandas-ta library.
    """
    def __init__(self, dataframe: pd.DataFrame, short_period=5, medium_period=10, upper_std=2, lower_std=1, tgt_hurdle=0.005, testsize=0.20):
        super().__init__(dataframe, testsize)
        self.short_period = short_period
        self.medium_period = medium_period
        self.upper_std = upper_std
        self.lower_std = lower_std
        self.tgt_hurdle = tgt_hurdle

    def create_all_features(self, fundamental_features=True, macro_features=True):
        if fundamental_features & macro_features:
            df_feat_ff = self.get_FA_features()
            df_feat_mc = self.get_Macro_features(df_feat_ff)
            df_comp_features = self.get_TA_features(df_feat_mc)

        elif fundamental_features & macro_features==False:
            df_feat_ff = self.get_FA_features()
            df_comp_features = self.get_TA_features(df_feat_ff)

        elif macro_features & fundamental_features==False:
            df_feat_mc = self.get_Macro_features(self.dataframe.copy())
            df_comp_features = self.get_TA_features(df_feat_mc)
        else:
            data = self.dataframe.copy()
            df_comp_features = self.get_TA_features(data)
        return df_comp_features


    # Create Features based on the given fundamentals of the company
    def get_FA_features(self):
        dt = self.dataframe.copy()
        threshold = 1e-5

        # Calculate Price-to-Earnings-to-Dividend Ratio (PED)
        ped_required_cols = ['PriceToEarnings', 'DividendYield']
        if all(col in dt.columns for col in ped_required_cols):
            dt['PED_Ratio'] = dt.apply(lambda row: row['PriceToEarnings'] / row['DividendYield']
            if abs(row['DividendYield']) > threshold else float('inf'), axis=1)
            logger.info("Fundamental Features: "
                        "Price-to-Earnings-to-Dividend Ratio (PED)Ratio feature successfully calculated")

        # Calculate Price to Earnings and Price to Book Combined (PEPB)
        pepb_required_cols = ['PriceToEarnings', 'PriceToBook']
        if all(col in dt.columns for col in pepb_required_cols):
            dt['PEPB_Ratio'] = (dt['PriceToEarnings'] + dt['PriceToBook']) / 2
            logger.info("Fundamental Features: "
                        "Price to Earnings and Price to Book Combined (PEPB)_Ratio feature successfully calculated")

        # Calculate Price to Cash & Price to Earnings Combined (PCFPER)
        pcfper_required_cols = ['PriceToEarnings', 'PriceToCash']
        if all(col in dt.columns for col in pcfper_required_cols):
            dt['PCFPER'] =  (dt['PriceToCash'] + dt['PriceToEarnings']) / 2
            logger.info("Fundamental Features: "
                        "Price to Cash & Price to Earnings Combined (PCFPER) feature successfully calculated")

        #Calculate combined valuation metric (cvm)
        cvm_required_cols = ['DividendYield', 'PriceToEarnings', 'PriceToCash', 'PriceToBook']
        if all(col in dt.columns for col in cvm_required_cols):
            dt['CVM'] = ((dt['DividendYield'] * 0.25) + (dt['PriceToEarnings'] * 0.25) +
                         (dt['PriceToBook'] * 0.25) + (dt['PriceToCash'] * 0.25))
            logger.info("Fundamental Features: "
                        "Combined Valuation Metric (CVM)_feature successfully calculated")

        else:
            dt = dt.copy()
            logger.info("No fundamental features were calculated")

        # Replace calculations with zero divisions which gives infinity values with zero
        cols_to_replace_inf = ['PED_Ratio', 'PEPB_Ratio', 'CVM', 'PCFPER']
        dt[cols_to_replace_inf] = dt[cols_to_replace_inf].replace([np.inf, -np.inf], 0)
        return dt

    #Create Features based on the given MacroEconomic Fundamentals
    def get_Macro_features(self, data):
        dt = data.copy()
        threshold = 1e-5 #thresholf for smallest possible divisor
        threshold2 = 1e-2

        # Create Yield Spread Feature
        yield_cols = ['2yrTreasury', '10yrTreasury']
        if all(col in dt.columns for col in yield_cols):
            dt['YieldSpread'] = (dt['10yrTreasury'] - dt['2yrTreasury'])  # Create Yield spread feature
            dt['TreasuryYieldRatio'] = dt.apply(lambda row: row['10yrTreasury'] / row['2yrTreasury']
            if abs(row['2yrTreasury']) > threshold else float('inf'), axis=1)  # Create Treasury Yield Feature,
            # converting very small values of denominator to inf and zerolized
            logger.info("Macro features: "
                        "Yield Spread feature successfully calculated")

        #Create CPI/GDP Growth Feature
        cpi_gdp_cols = ['CPI', 'GDP']
        if all(col in dt.columns for col in cpi_gdp_cols):
            dt['CPI_GDP_Ratio'] = dt.apply(lambda row: row['CPI'] / row['GDP']
            if abs(row['GDP']) > threshold else float('inf'),axis=1)  # Create CPI_GDP Ratio features,
            # converting very small values of denominator to inf and zerolized
            logger.info("Macro Features: "
                        "CPI/GDP Ratio feature successfully calculated")

        #Create CPI and Yield Correlation Feature
        cpi_yield_cols = ['CPI', '2yrTreasury']
        if all(col in dt.columns for col in cpi_yield_cols):
            dt['CPI_Yield_Correlation'] = dt['CPI'].rolling(window=22, min_periods=22).corr(
                dt['2yrTreasury'])  # Create CPI_yield Correlation Feature
            logger.info("Macro Features: "
                        "CPI vs Yield Correlation feature successfully calculated")

        #Real Interest Rates Feature
        real_rates_cols = ['CPI', '10yrTreasury']
        if all(col in dt.columns for col in real_rates_cols):
            dt['RealRates'] = dt.apply(lambda row: ((1 + row['10yrTreasury']) / (1 + row['CPI'])) - 1
            if abs(row['10yrTreasury']) > threshold2 else float('inf'), axis=1)  # Create Real Rates Features
            logger.info("Macro Features: "
                        "Real Interest Rates feature successfully calculated")

        cols_to_replace_inf = ['TreasuryYieldRatio', 'CPI_GDP_Ratio', 'RealRates']

        dt[cols_to_replace_inf] = dt[cols_to_replace_inf].replace([np.inf, -np.inf], 0)
        return dt

    #Create target variable (y)
    @staticmethod
    def get_y(data, short_prd, medium_prd, upper_std, lower_std, tgt):
        """
        Return dependent variable y, This return and volatility trade strategy play
        inputs includes, short_period number of days, medium_period number of days,
        upper_std length, lower_std length, and target outperformance before crossover is categorized as significant.
        """
        # Strategy 1 (returns and volatility play)
        data['shrt_prd_roll_rtn'] = data.Close.rolling(short_prd).mean() #5day returns
        data['medium_prd_roll_rtn'] = data.Close.rolling(medium_prd).mean() #15day returns
        data['shrt_prd_std'] = data.Close.rolling(short_prd).std() #Standard deviation of 5dayreturns
        data['medium_prd_std'] = data.Close.rolling(medium_prd).std() #standard deviation of 15day returns
        data['medium_prd_std_up'] = data['medium_prd_roll_rtn'] + (upper_std * data['medium_prd_std']) #Upper standard deviation of returns
        data['medium_prd_std_down'] = data['medium_prd_roll_rtn'] - (lower_std * data['medium_prd_std']) #down stanrdard deviation of returns

        data['predict'] = np.where(((data['shrt_prd_roll_rtn'] > ((1+tgt) * data['medium_prd_roll_rtn'])) &
                                    (data['shrt_prd_std'] <= data['medium_prd_std_up'])
                                    ), 1, 0) #trading strategy
        #Drop unwanted columns
        data.drop(['shrt_prd_roll_rtn', 'medium_prd_roll_rtn', 'shrt_prd_std', 'medium_prd_std', 'medium_prd_std_up',
                   'medium_prd_std_down'], inplace=True, axis=1)
        y = data['predict']
        return y

    # Create Technical Indicator Features
    def get_TA_features(self, data) -> pd.DataFrame:
        try:
            df = data.copy() #make a copy of the provided data

            df['days'] = df.index.day_name()# create days of the week feature

            # create all strategies from pandas-ta library.
            df.ta.study("All", lookahead=False, talib=False)
            # copy dataframe with technical indicators
            data = df.copy()
            #Define the target variable
            data['predict'] = self.get_y(data, self.short_period, self.medium_period,
                                         self.upper_std, self.lower_std, self.tgt_hurdle).values
            # drop unwanted features columns
            data.drop(
                ['QQEl_14_5_4.236', 'QQEs_14_5_4.236', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2',
                 'HILOs_13_21','HILOl_13_21', 'PSARr_0.02_0.2', 'SUPERTl_7_3.0', 'SUPERTs_7_3.0', 'SUPERTd_7_3.0',
                 'SUPERT_7_3.0', 'ZIGZAGs_5.0%_10', 'ZIGZAGv_5.0%_10', 'ZIGZAGd_5.0%_10', 'VIDYA_14', 'VHM_610'],
                axis=1, inplace=True)

            # drop #ohlcv data
            data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

            # drop columns with infinity as values within them
            specific_value = np.inf
            specific_value2 = -np.inf
            columns_to_drop = [col for col in data.columns if specific_value in data[col].values
                               or specific_value2 in data[col].values]
            data.drop(columns=columns_to_drop, axis=1, inplace=True)

            # drop the first set of 100rows because of features with initial observation window requiring over 100days of data
            df2 = data.copy()[100:]

            # backfill columns to address missing values
            df2 = df2.bfill(axis=1)
            logger.info("Technical Indicator Features: feature successfully calculated")
            return df2

        except:
            logger.error("TA_features not created successfully, check setup")


class FeaturesTransformation(FeaturesEngineering):
    """
    Features transformation class used to create a custom transformer which transforms the days column.
    It takes as input a dataframe format, with the weekday column clearly named 'days' and returns the
    transformation of the days column, based on a sine and cosine formula.

    Returns two columns 'dsin' and 'dcos' and drops the days column.
    """
    def __init__(self, dataframe: pd.DataFrame, testsize=0.20):
        super().__init__(dataframe, testsize)

    def transformDaysColumn(self):
        df = self.dataframe.copy()
        day_column = pd.DataFrame(df['days'])

        # Use customized DayTransfomer to transform days column
        dtrs = DayTransformer()
        day_transformed = dtrs.fit_transform(day_column)

        # drop days column from main dataframe and merge the transformed column
        df.drop(['days'], axis=1, inplace=True)
        # data_updated = pd.merge(df, day_transformed, on='Date', how='left')
        df = df.join(day_transformed, how='left')
        return df

class FeaturesSelection(FeaturesEngineering):
    """
    Feature selection class:
    - Accepts the dataframe of consisting of all the features.
    Methods includes:
    1. Wrapper method, which uses both Boruta and Recursive Filter Method to arrive at a smaller feature set
    relative to the provided features. It outputs a list of feature names selected by the wrapper method.
    2. Filter method, which uses correlation among features to select a smaller feature set.
    The filter method, uses as input the wrapper method features, and outputs a smaller set.
    """
    def __init__(self, dataframe: pd.DataFrame, testsize=0.20):
        super().__init__(dataframe, testsize)
        self.features_list = None

    def wrapper_boruta(self, df=None, max_iter=150, early_stopping=True, alpha=0.05, max_depth=5, verbose=0):
        if df is None:
            self.df_boruta = self.dataframe.copy()
        else:
            self.df_boruta = df

        self.class_weight = self.cwts(self.df_boruta)

        # Define Features - X and target/label - y
        # features column - X variable
        X = self.df_boruta.drop('predict', axis=1)
        # declare target column as y variable
        y = self.df_boruta['predict'].values.astype(int)  # convert the target values to integer

        self.features_list = self.df_boruta.drop('predict', axis=1).columns  #Column names

        X_train, X_test, y_train, y_test = self.traintestsplit(X, y)
        # convert to array
        X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

        # Feature selection, running the random forest classifier with all the created features.
        acc, f1score, class_rpt = self.randomforestSelection(X_train, X_test, y_train, y_test, self.class_weight)
        print(f"Pre-Boruta selection metrics: Accuracy Score: {acc: .2%}, f1_score: {f1score:.2%} \n")

        # Applying Boruta Selection
        # define Boruta feature selection method
        boruta_selector = BorutaPy(RandomForestClassifier(max_depth=max_depth, n_jobs=-1, class_weight=self.class_weight),
                                   n_estimators='auto', verbose=verbose, random_state=rnd_state(), max_iter=max_iter)

        # find all Boruta selected features from the feature set.
        boruta_selector.fit(X_train, y_train)
        self.boruta_selected_features = list(self.features_list[boruta_selector.support_])

        # call transform() on X to filter it down to selected features
        X_train_filtered = boruta_selector.transform(X_train)
        X_test_filtered = boruta_selector.transform(X_test)

        # Post Boruta selection, run the random forest classifier on the boruta selected features
        acc2, f1score2, class_rpt2 = self.randomforestSelection(X_train_filtered, X_test_filtered, y_train, y_test,
                                                                self.class_weight)
        print(f"\nUsing the ({len(self.boruta_selected_features)})Boruta"
              f"Selected Features, metrics: Accuracy Score: {acc2:.2%}, f1_score: {f1score2:.2%} \n")


        # RFE Features - use Recursive Feature Elimination (RFE) already created as a method to select features.
        # The number of features for the RFE to select is determined by the number of Boruta selected features.
        rfe_selected_features = self.wrapper_rfe(X_train, y_train)
        self.features_updated = list(set(self.boruta_selected_features) & set(rfe_selected_features))

        X_updated = self.df_boruta[self.features_updated]
        y_updated = y

        # updated features boruta and rfe
        X_train2, X_test2, y_train2, y_test2 = self.traintestsplit(X_updated, y_updated)

        # Post Boruta & RFE intersection evaluation
        acc3, f1score3, class_rpt3 = self.randomforestSelection(X_train2, X_test2, y_train2, y_test2, self.class_weight)
        print(
            f"\n Using Recursive Forward Elimination (RFE) approach to validate the "
            f"({len(self.boruta_selected_features)})features selected by Boruta approach,"
            f"({len(self.features_updated)})features which are the intersect features "
            f"for both Boruta and RFE, the evaluation metrics using the ({len(self.features_updated)})features are: "
            f"Accuracy Score: {acc3: .2%}, f1_score: {f1score3:.2%} \n")

        # Plot Boruta vs RFE selected features
        plot_data = pd.DataFrame({
            'Feature': list(set(self.boruta_selected_features + rfe_selected_features)),
            'Boruta_Pos': [1 if feature in self.boruta_selected_features else 0 for feature in
                                set(self.boruta_selected_features + rfe_selected_features)],
            'RFE_Pos': [1 if feature in rfe_selected_features else 0 for feature in
                             set(self.boruta_selected_features + rfe_selected_features)]
        })
        self.plot_intersected_features(plot_data)
        return self.features_updated


    #Recursive Forward Elimination (RFE)
    def wrapper_rfe(self, Xtrain, ytrain):
        rfe_clf = RFE(estimator=RandomForestClassifier(random_state=rnd_state(), class_weight=self.class_weight),
                      n_features_to_select=len(self.boruta_selected_features))
        rfe_clf.fit(Xtrain, ytrain)
        rfe_selected_features = self.features_list[rfe_clf.support_]
        return rfe_selected_features

    #Multicollinearity: filtering for correlation among features
    def filter_correlation(self, corr_coeff = 0.9, df=None):
        if df is None:
            df = self.df_boruta[self.features_updated]
            df['predict'] = self.df_boruta['predict'].values.astype(int)
        else:
            df = df
            df['predict'] =df['predict'].values.astype(int)

        # df['predict'] = self.df_boruta['predict'].values.astype(int)
        X_corr = df.drop('predict', axis=1)
        cls_weight = self.cwts(df)

        # Plot correlation matrix
        self.plot_correlation_matrix(X_corr)

        # get the list of correlated features based on the defined correlation threshold
        corr_features = self.correlated_features(X_corr, corr_coeff)

        # drop the correlated features from the updated feature set
        X_filtered = X_corr.drop(corr_features, axis=1)  # Alternatively

        # extract filtered feature names to a list.
        corr_features = X_filtered.columns.tolist()

        print(
            f"\n Solving for multicollinearity of features, and applying correlation coefficient of {corr_coeff}, "
            f"the ({len(self.features_updated)})features selected which are the intersected features of Boruta and Recursive Forward"
            f"Elimination (RFE) were filtered to {len(corr_features)}features \n")

        # print(corr_features)
        X_upd = df[corr_features]
        y_upd = df['predict'].values.astype('int')

        # updated filtered features run
        X_train3, X_test3, y_train3, y_test3 = self.traintestsplit(X_upd, y_upd)

        # Filtered selection results
        acc4, f1score4, class_rpt4 = self.randomforestSelection(X_train3, X_test3, y_train3, y_test3, cls_weight)
        print(
            f"\n After addressing the multicollinearity among features, applying RandomForestClassifier to "
            f"predictthe ({len(corr_features)})Filtered Features gives the following values for tracked metrics: "
            f"Accuracy Score: {acc4:.2%}, f1_score: {f1score4:.2%} \n")

        #tabulate final selected features
        df_filtered = pd.DataFrame({'Filtered Features Names': corr_features})
        # tab_index = [[i + 1] for i in range(0, len(df_filtered)+1)]
        df_filtered.insert(0, 'Index', range(1, len(df_filtered)+1))
        # header = ['#', 'Filtered Features Names']
        # print(tabulate(df_filtered, tablefmt="fancy_outline", headers=header))
        print(tabulate(df_filtered, tablefmt="fancy_outline", headers='keys', showindex=False))
        return corr_features


    @staticmethod
    def plot_intersected_features(df):
        # Assign random positions for each feature
        np.random.seed(42)  # For reproducibility
        df['Boruta_Pos'] = np.random.rand(len(df)) * 10  # Random values between 0 and 10
        df['RFE_Pos'] = np.random.rand(len(df)) * 10

        import plotly.express as px

        # Create scatter plot
        fig = px.scatter(
            df,
            x='Boruta_Pos',
            y='RFE_Pos',
            text='Feature',  # Display feature names
            title='Scatter Plot of Boruta vs RFE Selected Features',
            labels={'Boruta_Pos': 'Boruta Position', 'RFE_Pos': 'RFE Position'},
            size_max=10  # Maximum size for the markers
        )

        # Customize hover information and appearance
        fig.update_traces(textposition='top center', marker=dict(size=12, color='rgba(93, 164, 214, 0.6)',
                                                                 line=dict(width=2)))
        # Add gridlines and adjust layout
        fig.update_layout(
            xaxis=dict(showgrid=True, zeroline=True, showticklabels=False, title='Boruta Features',
                       color='black'),
            yaxis=dict(showgrid=True, zeroline=True, showticklabels=False, title='RFE Features',
                       color='black'),
            margin=dict(l=40, r=40, b=40, t=40), plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)'
        )
        # Show the plot
        fig.write_html(f"{getpath()}/Boruta_vs_RFE selected features plot_{datetime.now()}.html")
        fig.show()
