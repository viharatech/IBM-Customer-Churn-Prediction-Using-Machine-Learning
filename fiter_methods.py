import numpy as np
import pandas as pd
import sys
from logger import Logger
from sklearn.feature_selection import chi2, f_classif, f_regression, SelectKBest, mutual_info_classif, VarianceThreshold
from scipy.stats import ttest_ind, pearsonr, spearmanr
log = Logger.get_logs('filter_methods')

class FilterMethod:
    def chisquare(df, df1 , y):
        try:
            log.info(f"Befor shape:{df.shape}")
            selector = SelectKBest(score_func=chi2, k='all')
            x_new = selector.fit_transform(df, y)
            #Score card creation
            chi2_score = pd.DataFrame(
                {
                    'Features':df.columns,
                    'Chi_score': selector.scores_,
                    'p values':selector.pvalues_
                }
            ).sort_values(by='Chi_score', ascending=False)
            log.info(f'Chi score data frame :\n {chi2_score}')
            chi2_score = chi2_score[chi2_score['Features'] != 'sim']
            remove_features = chi2_score[chi2_score['p values'] > 0.05]['Features']
            df_filtered = df.drop(columns=remove_features)
            df_filtered1 = df1.drop(columns=remove_features)
            log.info(f"Removed columns: {list(remove_features)}")
            log.info(f"Befor shape:{df_filtered.shape}")
            return df_filtered, df_filtered1
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")

    def Anova_test(df,y):
        try:
            log.info(df.columns)
            selector1 =SelectKBest(score_func=f_classif, k='all')
            x_new1 = selector1.fit_transform(df, y)
            anova = pd.DataFrame(
                {
                    'features':df.columns,
                    'Anova':selector1.scores_,
                    'p_values':selector1.pvalues_
                }
            ).sort_values(by='Anova', ascending=False)

            log.info(anova)
            remove_feature1 = anova[anova['p_values'] >= 0.05]['features']
            log.info(f"Columns remove are: {list(remove_feature1)}")
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")

    def multi_classif(df, df1, y):
        try:
            m_selector = SelectKBest(score_func=mutual_info_classif, k='all')
            fl = m_selector.fit(df, y)
            multi_df = pd.DataFrame(
                {
                    'features': df.columns,
                    'Score': m_selector.scores_,
                }
            )
            log.info(multi_df)
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")

    def constant(x_train,x_test):
        try:
            con_v = VarianceThreshold(threshold=0)
            log.info('Constant filter Constant method is applied....')
            log.info(x_train)
            log.info(x_train.shape)
            con_v.fit(x_train)
            log.info(f'Columns in the train data are : {len(x_train.columns)} -> After applyin the constant columns with variance not 0 are : {sum(con_v.get_support())} and variance with 0 are : {sum(~con_v.get_support())}')
            x_train = x_train.drop(x_train.columns[~con_v.get_support()], axis=1)
            x_test = x_test.drop(x_test.columns[~con_v.get_support()], axis=1)
            log.info(f'Columns after droping are : {x_train.columns}')
            return x_train, x_test
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            log.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')

    def quasi_constant(x_train, x_test):
        try:
            qcon_v = VarianceThreshold(threshold=0.01)
            # qcon_v.fit(x_train,x_test)
            log.info('Constant filter Quasi Constant method is applied....')
            qcon_v.fit(x_train)
            log.info(f'Columns in the train data are : {len(x_train.columns)} -> After applyin the constant columns with variance not 0 are : {sum(qcon_v.get_support())} and variance with 0 are : {sum(~qcon_v.get_support())}')
            x_train = x_train.drop(x_train.columns[~qcon_v.get_support()], axis=1)
            x_test = x_test.drop(x_test.columns[~qcon_v.get_support()], axis=1)
            log.info(f'Columns after droping are : {x_train.columns}')
            return  x_train,x_test
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            log.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')

    def t_test_filter(df, df1, y):
        try:
            selected = []
            for i in df.columns:
                g1 = df[y==0][i]
                g2 = df[y==1][i]
                if g1.empty or g2.empty:
                    log.warning(f"{i}: empty groups, check y alignment.")
                    continue

                t, p = ttest_ind(g1, g2, nan_policy='omit')
                log.info(f"T={t:.3f}, P={p:.5f}")
                if p < 0.05:
                    selected.append(i)

            log.info(f"Selected Features: {selected}")
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            log.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')

    def co_relation(df,df1, y):
        try:
            # log.info(y)
            # log.info(y.isnull().sum())
            # y = y.map({'Yes':1, 'No':0}).astype(int)
            features_significant = []
            for i in df.columns:
                r, p = pearsonr(df[i], y)
                log.info(f'{i}----->{r}')
                if p < 0.05:
                    features_significant.append(i)
            log.info(f'{features_significant}')
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            log.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')