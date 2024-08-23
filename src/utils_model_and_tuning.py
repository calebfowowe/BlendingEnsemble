# LIBRARIES
# Data manipulation libraries
import pandas as pd
import numpy as np
from datetime import datetime

# Data visualization Library
import plotly.express as px
import plotly.graph_objects as go

px.height, px.width = 500, 700

# sklearn libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

#Inbuilt library
from functools import partial

#Hyperparameter tuning module
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING) #setting verbosity to zero
from tabulate import tabulate

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Other external modules used for the Backtest
import quantstats as qs
from backtesting import Backtest, Strategy

from src.utils_data_processing import getpath, rnd_state


#BLENDED ENSEMBLE
class Blending:
    """
    Blending ensemble class, used for implementing Blending ensemble model.
    Instantiation parameters includes:
    - X : Features
    - y : Labels/Target
    - basemodels: dictionary of all the basemodels, with keys which are model short_name and values which is also
    another dictionary of model parameters in key-value pairs.
    - Metamodel: a variable containing the dictionary of metamodel parameters

    """
    def __init__(self, X, y, basemodels, metamodel, testsize=0.20, valsize=0.20, scaler=StandardScaler()):
        self.X = X
        self.y = y
        self.models = basemodels
        self.testsize = testsize
        self.metamodel = metamodel
        self.valsize = valsize
        self.scaler = scaler

        # apply split - full training and test set
        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(self.X, self.y,
                                                                                          test_size=self.testsize,
                                                                                          shuffle=False)
        # apply further split on training set - training and validation set
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_full, self.y_train_full,
                                                                              test_size=self.valsize, shuffle=False)

    # evaluate predictions
    def get_accuracy(self, y_pred):
        score = accuracy_score(self.y_test, y_pred)
        print(f"Blender Accuracy: {score * 100:.1}%")

    def get_f1score(self, y_pred):
        score = f1_score(self.y_test, y_pred, average='weighted')
        print(f"Blender f1score: {score * 100:.1}%")

    def plot_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        labels = ['Class 0', 'Class 1']
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            texttemplate="%{z}", textfont={"size":20}
        ))
        fig.update_layout(
            title="Confusion matrix",
            xaxis={"title":"Predicted label", 'showgrid': False},
            yaxis={"title": "True label", 'showgrid': False},
            width=700,
            height=700,
        )
        fig.show()

    def plot_roc(self, y_prob):
        fpr, tpr, thresholds = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        #create the ROC curve plot
        fig = go.Figure()

        # Add ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'Classifier (AUC = {roc_auc:.2f})',
            line=dict(color='red', width=2),
        ))

        # Add the diagonal line representing random chance (no skill classifier)
        fig.add_trace(go.Scatter(
            x=[0.0,1], y=[0.0,1],
            mode='lines',
            name='Random 50:50',
            line=dict(color='navy', width=2, dash='dash')
        ))

        #Update layout
        fig.update_layout(
            title='Receiver Operating Characteristics (ROC) curve',
            xaxis={'title': 'False Positive Rate', 'scaleanchor': "x", 'scaleratio': 1},
            yaxis={'title': 'True Positive Rate', 'scaleanchor': "y", 'scaleratio': 1},
            width=700,
            height=700,
            legend={
                'orientation': 'h', 'yanchor': 'bottom', 'y': 0.1, 'xanchor': 'center', 'x': 0.5
            }
        )
        #Dipslay plot
        fig.show()

    # classification report
    def get_classification_report(self, y_pred):
        print("Classification Report")
        print(classification_report(self.y_test, y_pred))

    def get_classification_report_full(self, y_full_pred):
        print("Classification Report")
        print(classification_report(self.y, y_full_pred))


   #fit/Train the ensemble model
    def fit_blended_ensemble(self):
        #fit all models on the training set and predict on hold out set
        meta_X = list()
        for name, mod in self.models.items():
            mod = Pipeline([
                ('scaler', self.scaler),
                ('classifier', mod),
            ])
            #fit in training set
            mod.fit(self.X_train, self.y_train)
            #predict the base model on the validation set
            yhat = mod.predict_proba(self.X_val)[:,1]
            #store predictions as input for blending (meta learner)
            meta_X.append(yhat)
        #create 2d array from predictions, ech set is an input feature
        meta_X = np.column_stack(meta_X)

        # define blending model
        blender = self.metamodel
        #fit the belnder (meta model) using the stacked predictions from the base models as features
        blender.fit(meta_X, self.y_val)
        self.blender = blender
        return self.blender

    #make a prediction with the blending ensemble
    def predict_blended_ensemble(self):
        #make predictions with base models
        meta_X = list() # To store the basemodels predictions of the test data
        y_full = list() #To store the basemodels predictions of the full dataset
        for name, mod in self.models.items():
            #predict with base model
            mod = Pipeline([
                ('scaler', self.scaler),
                ('classifier', mod),
            ])
            yhat = mod.predict_proba(self.X_test)[:,1] #interim prediction the test set on the basemodel
            yhatfull = mod.predict_proba(self.X)[:,1] #interim prediction of the full dataset on the basemodel
            #Append predictions
            meta_X.append(yhat)
            y_full.append(yhatfull)

        meta_X = np.column_stack(meta_X) #stack test set prediction
        y_full = np.column_stack(y_full) #stac full data predictions

        y_pred = self.blender.predict(meta_X) #final prediction of the test dataset using metamodel
        y_prob = self.blender.predict_proba(meta_X)[:,-1] #final predict proba of the test dataset using metamodel
        y_full = self.blender.predict(y_full) #final prediction of the full dataset using metamodel

        # Calculate scores of interest
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        return acc, f1, y_pred, y_prob, y_full

    def runBlendingEnsemble(self):
        """
        Function to run the blending ensemble includes fitting and predicting both on the test set and the
        full dataset and produces scores and predictions
        :return: accuracy score, f1score, y_predictions, y_probabilities, y_full_predictions
        It also prints the plots
        """
        # fit blending ensemble to generate the fitted blender
        ensemble = self.fit_blended_ensemble()
        # predict
        acc, f1score, ypred, yprob, yfull = self.predict_blended_ensemble()

        # plot roc, confusion matrix and generate classification report.
        self.plot_confusion_matrix(ypred)
        print('')
        self.plot_roc(yprob)
        self.get_classification_report(ypred)
        # self.get_classification_report_full(yfull)

        return acc, f1score, ypred, yprob, yfull


# HYPERPARAMETER TUNING
class HpTuning:
    """
    Hyperparameter tuning of the basemodels and metamodels. Models are trained independently.
    Tuning focuses on optimizing two scores, i.e. two objectives, accuracy score and f1score.
    Input data includes X, y, variables (i.e. Features and target respectively).
    Optional inputs:
     - number of splits (n_splits) to be used in TimeSeries cross-validation split, default is 4
     - number of trials (n_trials), default is 10
    """
    def __init__(self, X, y, n_splits=4, n_trials=10, seed=rnd_state()):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.sampler = optuna.samplers.TPESampler(seed=seed)  # setting random seed inorder to make results reproducible.

    def optim(self, model, directions):
        study = optuna.create_study(study_name='models_hp', directions=directions, sampler=self.sampler)
        optimizing_function = partial(model, x=self.X, y=self.y)
        study.optimize(optimizing_function, n_trials=self.n_trials)
        trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])

        return trial_with_highest_accuracy

    ### Tuning time split
    def optuna_tscv(self, x, y, model):
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=1)
        f1score = []
        acc_score = []

        # Fit model for every split
        for idx in tscv.split(x):
            train_idx, test_idx = idx[0], idx[1]
            xtrain, ytrain = x[train_idx], y[train_idx]
            xtest, ytest = x[test_idx], y[test_idx]
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", model)
            ])
            clf.fit(xtrain, ytrain)
            preds = clf.predict(xtest)

            cv_f1 = f1_score(ytest, preds, average='weighted')
            cv_acc = accuracy_score(ytest, preds)

            f1score.append(cv_f1)
            acc_score.append(cv_acc)
        return np.mean(f1score), np.mean(acc_score)

    # DECISION TREE CLASSIFIER
    def dt_objective(self, trial, x, y):
        # Parameters
        max_depth = trial.suggest_int("max_depth", 1, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10, log=True)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5, log=True)
        ccp_alpha = trial.suggest_float("ccp_alpha", 0.01, 1.0, log=True)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                ccp_alpha=ccp_alpha,
                criterion=criterion,
                random_state=rnd_state()
            ))
        ])
        f1score, acc_score = self.optuna_tscv(x=x, y=y, model=model)
        return f1score, acc_score

    def optimize_dt(self, directions=['maximize', 'maximize']):
        optimal = self.optim(self.dt_objective, directions=directions)
        return optimal

    # XGB CLASSIFIER
    def xgb_objective(self, trial, x, y):
        # Parameters
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        max_depth = trial.suggest_int('max_depth', 3, 30)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        lambda1 = trial.suggest_float('lambda', 0, 10)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        alpha = trial.suggest_float('alpha', 0, 10)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        tree_method = trial.suggest_categorical("tree_method", ["hist", "exact", "approx", "auto"])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                lambda_=lambda1,
                min_child_weight=min_child_weight,
                alpha=alpha,
                colsample_bytree=colsample_bytree,
                subsample=subsample,
                tree_method=tree_method,
                random_state=rnd_state()
            ))
        ])
        f1score, acc_score = self.optuna_tscv(x=x, y=y, model=model)
        return f1score, acc_score

    def optimize_xgb(self, directions=['maximize','maximize']):
        optimal = self.optim(self.xgb_objective, directions=directions)
        return optimal

    def optimize_ada(self, directions=['maximize','maximize']):
        optimal = self.optim(self.ada_objective, directions=directions)
        return optimal

    # SVC CLASSIFIER
    def svc_objective(self, trial, x, y):
        # Parameters
        C = trial.suggest_float("C", 0.001, 10, log=True)
        tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)
        gamma = trial.suggest_categorical("gamma", ['scale', 'auto'])
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        degree = trial.suggest_int("degree", 1, 5)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                tol=tol,
                random_state=rnd_state()
            ))
        ])
        f1score, acc_score = self.optuna_tscv(x=x, y=y, model=model)
        return f1score, acc_score

    def optimize_svc(self, directions=['maximize','maximize']):
        optimal = self.optim(self.svc_objective, directions=directions)
        return optimal

    # K-Nearest Neigbhours (KNN) CLASSIFIER
    def knn_objective(self, trial, x, y):
        # Parameters
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
        metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
        leaf_size = trial.suggest_int("leaf_size", 10, 100)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                weights=weights,
                metric=metric,
                n_jobs=-1,
                n_neighbors=n_neighbors,
                leaf_size=leaf_size
            ))
        ])

        f1score, acc_score = self.optuna_tscv(x=x, y=y, model=model)
        return f1score, acc_score

    def optimize_knn(self, directions=['maximize','maximize']):
        optimal = self.optim(self.knn_objective, directions=directions)
        return optimal

    # LOGISTIC REGRESSION
    def lr_objective(self, trial, x, y):
        # Parameters
        C = trial.suggest_float("C", 0.001, 1, log=True)
        tol = trial.suggest_float("tol", 0.001, 0.01, log=True)
        solver = trial.suggest_categorical("solver", ['lbfgs', 'liblinear', 'newton-cholesky', 'sag', 'saga'])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=C,
                tol=tol,
                solver=solver,
                n_jobs=-1,
                random_state=rnd_state()
            ))
        ])
        f1score, acc_score = self.optuna_tscv(x=x, y=y, model=model)
        return f1score, acc_score

    def optimize_lr(self, directions=['maximize', 'maximize']):
        optimal = self.optim(self.lr_objective, directions=directions)
        return optimal

    # GAUSSIAN NB()
    def bayes_objective(self, trial, x, y):
        # Parameters
        n_estimators = trial.suggest_int("n_estimators", 100, 1500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        var_smoothing = trial.suggest_float("var_smoothing", 0.01, 1, log=True)


        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB(
                        var_smoothing= var_smoothing
            ))
        ])
        f1score, acc_score = self.optuna_tscv(x=x, y=y, model=model)
        return f1score, acc_score

    def optimize_bayes(self, directions=['maximize', 'maximize']):
        optimal = self.optim(self.bayes_objective, directions=directions)
        return optimal


    # ADABOOST
    def ada_objective(self, trial, x, y):
        # Parameters
        n_estimators = trial.suggest_int("n_estimators", 50, 1000)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 3.0, log=True)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate
            ))
        ])
        f1score, acc_score = self.optuna_tscv(x=x, y=y, model=model)
        return f1score, acc_score


    # RANDOM-FOREST CLASSIFIER
    def rf_objective(self, trial, x, y):
        # Parameters
        n_estimators = trial.suggest_int("n_estimators", 100, 1500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        max_features = trial.suggest_float("max_features", 0.01, 1)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10, log=True)
        ccp_alpha = trial.suggest_float("ccp_alpha", 0.01, 1.0, log=True)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=min_samples_split,
                ccp_alpha=ccp_alpha,
                random_state=rnd_state()
            ))
        ])
        f1score, acc_score = self.optuna_tscv(x=x, y=y, model=model)
        return f1score, acc_score

    def optimize_rf(self, directions=['maximize', 'maximize']):
        optimal = self.optim(self.rf_objective, directions=directions)
        return optimal

    @staticmethod
    def hp_preview(params_list, params_name):
        for i, param in enumerate(params_list):

            tuned_params = pd.DataFrame(list(param.items()), columns=['Hyper-parameter', 'Tuned Values'])
            tuned_params.insert(0, 'Index', range(1, len(param) + 1))
            print(params_name[i])
            print(tabulate(tuned_params, tablefmt="grid", headers='keys', showindex=False))



#BACKTESTING
class SimpleBacktest:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @staticmethod
    def sharpe_ratios(data):
        bts = data[['Benchmark', 'Strategy']]
        print(qs.stats.sharpe(bts))

    def html_report(self, company_name=None):
        date_time = datetime.now().strftime('%Y-%m-%d, %H%M%S')
        if company_name is not None:
            # generate report and save in the output folder
            qs.reports.html(self.bt_data['Strategy'], self.bt_data['Benchmark'],
                            title=f'Strategy BackTest Report for {company_name}',
                            output=f'{getpath()}/{company_name}_backtesting_report_test_period-{date_time}.html')
        else:
            qs.reports.html(self.bt_data['Strategy'], self.bt_data['Benchmark'],
                            title=f'Strategy BackTest Report',
                            output=f'{getpath()}/backtesting_report_test_period-{date_time}.html')

    def show_report(self, company_name=None):
        if company_name is not None:
            report = qs.reports.full(self.bt_data['Strategy'], benchmark=self.bt_dat['Benchmark'], mode='full',
                            title=f'Strategy BackTest Report for {company_name}')
        else:
            report = qs.reports.full(self.bt_data['Strategy'], benchmark=self.bt_dat['Benchmark'], mode='full',
                            title=f'Strategy BackTest Report')
        return report


    def approach1(self, label, horizon):
        df = self.dataframe.copy()
        # Extract Close prices over the range of dates of the full model
        backtest_data = df[['Close', 'Open']][-len(label):]
        backtest_data['Signal'] = label

        # Entry logic
        backtest_data['Entry'] = np.where(backtest_data['Signal'] == 1, backtest_data['Close'],
                                          0)  # when the strategy signal is 1, we enter into a trade,
                                                # and buy at the end of day's close.
        # Exit Logic
        backtest_data['Exit'] = np.where((backtest_data['Entry'] != 0) &
                                         (backtest_data['Open'].shift(-horizon) <= backtest_data['Close']),
                                         backtest_data['Open'].shift(-horizon), 0)  #
        backtest_data['Exit'] = np.where((backtest_data['Entry'] != 0) &
                                         (backtest_data['Open'].shift(-horizon) > backtest_data['Close']),
                                         backtest_data['Close'].shift(-horizon), backtest_data['Exit'])

        # Calculate MTM
        backtest_data['P&L'] = backtest_data['Exit'] - backtest_data['Entry']

        # Generate Equity Curve
        backtest_data['Equity'] = backtest_data['P&L'].cumsum() + backtest_data['Close'][0]

        # Calculate Benchmark Return
        backtest_data['Benchmark'] = np.log(backtest_data['Close']).diff().fillna(0)

        # Calculate Strategy Return
        backtest_data['Strategy'] = (backtest_data['Equity'] / backtest_data['Equity'].shift(horizon) - 1).fillna(0)
        backtest_data = backtest_data.iloc[:-1]
        #Extract backtest data into a dataframe
        self.bt_data = backtest_data[['Benchmark', 'Strategy']]

        return backtest_data #returns a dataframe with the calculated trading parameters.


class SignalStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        current_signal = self.data.Signal[-1]
        if current_signal == 1:
            if not self.position:
                self.buy()
        else:
            if self.position:
                self.position.close()
        pass

class Btest:
    """
    Class to run the ML generated trade signals derived through the popular Backtetsing module
    Input includes:
        - Dataframe use in the analysis, consisting of Open, High, Low, Close, and Volume data
        - The trend signal generated in a list format
    Using the provided input, strategy is run through the method runStrategy()
    Outputs:
        - statistics vs Buy hold & strategy: through method: runstats()
        - plot of the buy sell signals: through the method: plotstats()
    """
    def __init__(self, dataframe, signal, commission=0.002, exclusive=True, starting_cash=10_000):
        self.dataframe = dataframe
        self.signal = signal
        self.commission = commission
        self.exclusive = exclusive
        self.starting_cash = starting_cash

    def runStrategy(self):
        data = self.dataframe[['Open', 'High', 'Low', 'Close']][-len(self.signal):]
        data['Signal'] = self.signal

        self.bt = Backtest(data, SignalStrategy, cash=self.starting_cash,
                           commission=self.commission, exclusive_orders=self.exclusive) #use the backtesting class
        return self.bt
    def runstats(self):
        return self.bt.run()
    def plotstats(self):
        return self.bt.plot()



