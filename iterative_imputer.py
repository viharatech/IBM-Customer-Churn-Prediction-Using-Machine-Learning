from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import sys
from logger import Logger

log = Logger.get_logs('iterative_imputer')


class Iterative:
    def it_imp(df):
        try:
            log.info('Iterative imputer started...')
            imputer = IterativeImputer(estimator=DecisionTreeRegressor(),max_iter=10, initial_strategy='mean')
            log.info(f'Checking the null values :-{df.isnull().sum()}')
            df_imputed = imputer.fit_transform(df)

            df1 = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)

            log.info(f"Imputation complete. Shape: {df1.shape}")
            log.info(f"Missing values after imputation:\n{df1.isnull().sum()}")
            # log.info(df1[df['TotalCharges']==pd.Na])
            # log.info(df1.isnull().sum())
            return df1

        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')
