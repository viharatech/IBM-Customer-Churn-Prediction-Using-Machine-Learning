import pandas as pd
from sklearn.impute import KNNImputer
from logger import Logger
log = Logger.get_logs('knn_imputer')
import sys
import warnings
warnings.filterwarnings('ignore')
class KNNImpute:
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