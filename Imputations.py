import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from feature_engine.imputation import MeanMedianImputer, ArbitraryNumberImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from logger import Logger
log = Logger.get_logs('RandomSampleImputation')
class Misssing:
    def RandomSampleImputation(X_train, X_test):
        try:
            columns = []
            log.info("RAndom sample has started ............")
            for i in X_train.columns:
                if X_train[i].isnull().sum() !=0:
                    columns.append(i)
                    log.info(i)
                    X_train_values = X_train[i].dropna().sample(X_train[i].isnull().sum(), random_state=42)
                    X_test_values = X_test[i].dropna().sample(X_test[i].isnull().sum(), random_state=42)
                    log.info(f'Values {X_train_values}')
                    X_test_values.index = X_test[X_test[i].isnull()].index
                    X_train_values.index = X_train[X_train[i].isnull()].index

                    X_train[i+'replaced'] = X_train[i].copy()
                    X_test[i+'replaced'] = X_test[i].copy()

                    X_train.loc[X_train[i].isnull(), i+'replaced'] = X_train_values
                    X_test.loc[X_test[i].isnull(), i+'replaced'] = X_test_values
                    log.info(X_train.isnull().sum())
                    log.info(X_test.isnull().sum())
            return X_train, X_test

        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    def Mean_medianImputer(X):
        try:
            imputer1 = MeanMedianImputer()
            check_1 = imputer1.fit_transform(X)
            df1 = pd.DataFrame(check_1, columns=X.columns)
            log.info(df1.isnull().sum())
            log.info("Done")
            return  df1
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    def SimpleImput(X, st):
        try:
            imputer2 = SimpleImputer(strategy=st)
            df_num = X.select_dtypes(['int64', 'float64'])
            columns1 = []
            for i in df_num.columns:
                if df_num[i].dtype !=object:
                    if df_num[i].isnull().sum() !=0:
                        columns1.append(i)

            values = imputer2.fit_transform(X[columns1])
            df2 = pd.DataFrame(values, columns=columns1, index=X.index)
            log.info(df2)
            log.info(columns1)
            log.info(f"SimpleImputer{df2.isnull().sum()}")
            return df2
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    def ArbitoryImputation(Y):
        try:
            imputer3 = ArbitraryNumberImputer()
            df_num = Y.select_dtypes(['int64', 'float64'])
            columns1 = []
            for i in df_num.columns:
                if df_num[i].dtype != object:
                    if df_num[i].isnull().sum() != 0:
                        columns1.append(i)

            values = imputer3.fit_transform(Y[columns1])
            df3 = pd.DataFrame(values, columns=columns1)
            log.info(df3)
            log.info(columns1)
            log.info(df3.isnull().sum())
            return df3

        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    def it_imp(df):
        try:
            log.info('Iterative imputer started...')
            imputer = IterativeImputer(estimator=DecisionTreeRegressor(),max_iter=10, initial_strategy='mean')
            log.info(f'Checking the null values :-{df.isnull().sum()}')
            df_imputed = imputer.fit_transform(df)
            log.info(f"It check{df.columns}")
            df1 = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)

            log.info(f"Imputation complete. Shape: {df1.shape}")
            log.info(f"Missing values after imputation:\n{df1.isnull().sum()}")
            # log.info(df1[df['TotalCharges']==pd.Na])
            # log.info(df1.isnull().sum())
            return df1

        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    def knn_im(df):
        try:
            log.info('KNN imputer started...')
            imputer = KNNImputer(n_neighbors=9, weights='distance', metric='nan_euclidean')
            df_imputed = imputer.fit_transform(df.values.reshape(-1, 1))  # ensure correct shape
            df1 = pd.DataFrame(df_imputed, columns=['Total_charges_knn'], index=df.index)
            log.info(f"Imputation complete. Shape: {df1.shape}")
            log.info(df1.isnull().sum())
            log.info(df1.head())
            return df1
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')