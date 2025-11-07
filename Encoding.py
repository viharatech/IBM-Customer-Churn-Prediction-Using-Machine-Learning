import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import sys
from logger import Logger
log = Logger.get_logs('Encode')
class Encoding:
    def LabelEncoding_calumns(df):
        try:
            log.info(df)
            log.info(df)
            # log.info(df.columns)
            lb = LabelEncoder()
            ft = lb.fit_transform(df)
            # df = df.drop(remove, axis=1)
            # log.info(df.columns)
            return ft
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')
    def Onehot_encode(df, nominal):
        try:
            encoder1 = OneHotEncoder(drop='first', sparse_output=False)
            encode_array = encoder1.fit_transform(df[nominal])
            encoded_col_names = encoder1.get_feature_names_out(nominal)
            encode_df_main = pd.DataFrame(encode_array, columns=encoded_col_names, index=df.index)
            df1 = pd.concat([df.drop(nominal, axis=1), encode_df_main], axis=1)
            log.info(f"One-hot encoded columns: {nominal}")
            log.info(f"Resulting shape: {df1.shape}")
            log.info(df1.info())
            df1 = df1.drop(['Contract'], axis=1)
            return df1
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    def Ordinal_encoder(df,cols):
        try:
            encoder3 = OrdinalEncoder()
            log.info(df.columns)
            encode_df = encoder3.fit_transform(df[cols])
            df = pd.DataFrame(encode_df, columns=cols, index=df.index)
            log.info(df.shape)
            log.info(df)
            return df
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')