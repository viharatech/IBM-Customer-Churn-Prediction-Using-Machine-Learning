import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from feature_engine.outliers import Winsorizer, OutlierTrimmer
from logger import Logger
from visual import plot_kde_comparison
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
log = Logger.get_logs('Outliers')
class Outlier:
    def Winzos(df, method, fold):
        try:
            log.info(f"{df.shape}")
            log.info("Started")
            tf = Winsorizer(capping_method=method, tail='both',fold=fold)
            df_o1 = tf.fit_transform(df)
            log.info('Done')
            log.info(df_o1.head(5))

            # plot_kde_comparison(df_o1, '_qt', 'r', f'Outlier-{method}' )

            log.info(f'{df_o1.shape}')
            return df_o1
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    # def Outlier_trim(self): Data is lossing
    #     try:
    #         log.info("Outlier_trim has started................")
    #         tf1 = OutlierTrimmer()
    #     except Exception:
    #         exc_type, exc_msg, exc_tb = sys.exc_info()
    #         log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    def Isolation_forests(df):
        try:
            log.info(f'Befor checking the shape:{df.shape}')
            log.info("IsolationForest with capping has started......")
            df_num = df.select_dtypes(include=['number']).copy()
            iso = IsolationForest(contamination=0.1, random_state=42)
            df_num['Outlier'] = iso.fit_predict(df_num)

            for col in df_num.columns:
                if col != 'Outlier':
                    # Calculate IQR limits
                    Q1 = df_num[col].quantile(0.25)
                    Q3 = df_num[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_limit = Q1 - 1.5 * IQR
                    upper_limit = Q3 + 1.5 * IQR
                    # Handle only detected outliers (-1)
                    for idx in df_num[df_num['Outlier'] == -1].index:
                        # log.info(idx)
                        value = df_num.loc[idx, col]
                        # log.info(value)
                        # If value < lower limit → set to lower limit
                        if value < lower_limit:
                            df_num.loc[idx, col] = lower_limit
                        # If value > upper limit → set to upper limit
                        elif value > upper_limit:
                            df_num.loc[idx, col] = upper_limit
                        # else: keep as is

            
            df_num = df_num.drop(columns=['Outlier'])
            df[df_num.columns] = df_num[df_num.columns]
            log.info(f'After checking the shape:{df.shape}')
            log.info("IsolationForest capping completed successfully.")
            # plot_kde_comparison(df, '_qt', 'r', f'Outlier-Isolation')
            return df
        except Exception as e:
            log.error(f"Error in IsolationForest: {str(e)}")

    def percentail_cap(df, lower_per=0.01, upper_per=0.99):
        try:
            df = df.copy()
            for col in df.columns:
                lower = df[col].quantile(lower_per)
                upper= df[col].quantile(upper_per)
                df[col] = df[col].clip(lower, upper)
            plot_kde_comparison(df, '_qt', 'r', f'Outlier-percentile')
        except Exception as e:
            log.error(f"Error in IsolationForest: {str(e)}")
            return df


