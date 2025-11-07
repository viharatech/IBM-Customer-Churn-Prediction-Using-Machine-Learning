import numpy as np
import pandas as pd
from sklearn.preprocessing import power_transform, quantile_transform, QuantileTransformer
from logger import Logger
import sys

log = Logger.get_logs('VariableTransformations')

class VariableTransformations:

    def power_transform_check(df, method):
        try:
            df_num = df.select_dtypes(include=['number']).copy()
            transformed = power_transform(df_num, method=method)
            df_pt = pd.DataFrame(transformed, columns=[col + '_pt' for col in df_num.columns], index=df.index)
            df_out = pd.concat([df, df_pt], axis=1)
            log.info("Power transformation completed successfully.")
            return df_out
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")
            return df


    def quantile_transform_check(df):
        try:
            df_num = df.select_dtypes(include=['number']).copy()
            transformed = quantile_transform(df_num, output_distribution='normal', random_state=42)
            df_qt = pd.DataFrame(transformed, columns=[col + '_qt' for col in df_num.columns], index=df.index)
            df_out = pd.concat([df, df_qt], axis=1)
            log.info("Quantile transformation completed successfully.")
            log.info(df_out.isnull().sum())
            log.info(df_out.head(10))
            return df_out
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")

    def sqrt_transform_check(df):
        try:
            df_num = df.select_dtypes(include=['number']).copy()

            for i in df_num.columns:
                df_num[i+'_sqrt'] = np.sqrt(df_num[i])

            log.info("Square Root transformation completed successfully.")
            log.info(np.isfinite(df_num))
            return df_num

        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")
            return df

    def log_transform_check(df):
        try:
            df_num = df.select_dtypes(include=['number']).copy()

            for i in df_num.columns:
                df_num[i+'_log'] = np.log(df_num[i] + 1)
            # df_log = pd.DataFrame(, columns=[col + '_log' for col in df_num.columns], index=df.index)
            # df_out = pd.concat([df, df_log], axis=1)
            log.info(df_num.columns)
            log.info(type(df_num))
            log.info("Log(+1) transformation completed successfully.")
            log.info(np.isfinite(df_num))
            return df_num

        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")
            return df

    def exp_transform_check(df):
        try:
            df_num = df.select_dtypes(include=['number']).copy()
            log.info(df_num.isnull().sum())
            for i in df_num.columns:
                df_num[i+'_exp'] = np.exp(df_num[i] + 1)
            # df_log = pd.DataFrame(, columns=[col + '_log' for col in df_num.columns], index=df.index)
            # df_out = pd.concat([df, df_log], axis=1)
            log.info(df_num.columns)
            log.info(type(df_num))
            log.info(np.isfinite(df_num).sum())
            log.info("exp transformation completed successfully.")
            return df_num

        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")
            return df

    def cbrt_transform_check(df):
        try:
            df_num = df.select_dtypes(include=['number']).copy()

            for i in df_num.columns:
                df_num[i+'_cbrt'] = np.cbrt(df_num[i])

            log.info("Cube root Root transformation completed successfully.")
            return df_num

        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")
            return df


    def tanh_transform_check(df):
        try:
            df_num = df.select_dtypes(include=['number']).copy()

            for i in df_num.columns:
                df_num[i+'_tanh'] = np.tanh(df_num[i])

            log.info("Tanh root Root transformation completed successfully.")
            return df_num

        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")
            return df

    def quantile_rank_transform_check(df):
        try:
            df_num = df.select_dtypes(include=['number']).copy()
            transformed = QuantileTransformer(output_distribution='normal', random_state=42)
            transformed = transformed.fit_transform(df_num)
            df_qtr = pd.DataFrame(transformed, columns=[col + '_qtr' for col in df_num.columns], index=df.index)
            df_out = pd.concat([df, df_qtr], axis=1)
            log.info("Quantile rank transformation completed successfully.")
            return df_out
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")

    def power_trans(df):
        try:
            df_num = df.select_dtypes(include=['number']).copy()
            for i in df_num.columns:
                df_num[i + '_tsqrt'] = 1/np.sqrt(df_num[i])
            return df_num
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")
