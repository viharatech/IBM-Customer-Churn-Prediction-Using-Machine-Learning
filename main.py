import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings
from sklearn.metrics import accuracy_score
from logger import Logger
from sklearn.model_selection import train_test_split
from Imputations import Misssing
from variable_transformations import VariableTransformations
from visual import plot_kde_comparison
from outliers import Outlier
from fiter_methods import FilterMethod
from Encoding import Encoding
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from model import common
from check_model import che
from sklearn.linear_model import LogisticRegression
import pickle
warnings.filterwarnings('ignore')
log = Logger.get_logs('main')

class Churn:
    def __init__(self):
        try:
            self.data = pd.read_csv('C:\\Users\\sivan\\OneDrive - MSFT\\Intership\\Data\\churn_dataset.csv')
            log.info(f"Data loaded with shape {self.data.shape}")
            self.data['TotalCharges'] = self.data['TotalCharges'].replace(['', ' '], pd.NA)
            self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'])
            self.data = self.data.drop('customerID', axis=1)
            X = self.data.drop('Churn', axis=1)
            y = self.data['Churn']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            log.info(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    def handling_missing_values(self):
        try:
            a = self.X_train.select_dtypes(['int64', 'float64'])
            b = self.X_test.select_dtypes(['int64', 'float64'])
            imputed_df2 = Misssing.it_imp(a)
            imputed_df3 = Misssing.it_imp(b)
            self.X_train['TotalCharges_itimp'] = imputed_df2['TotalCharges']
            self.X_train['MonthlyCharges_itimp'] = imputed_df2['MonthlyCharges']
            self.X_test['TotalCharges_itimp'] = imputed_df3['TotalCharges']
            self.X_test['MonthlyCharges_itimp'] = imputed_df3['MonthlyCharges']
            # self.X_train, self.X_test = Misssing.RandomSampleImputation(self.X_train, self.X_test)
            self.X_train.to_csv('check.csv', index=False)
            log.info("Data saved successfully after imputations")
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    def visual_check(self):
        try:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            self.X_train['TotalCharges'].plot(kind='kde', color='r', label=f'Original ({self.X_train["TotalCharges"].std():.2f})')
            self.X_train['TotalCharges_itimp'].plot(kind='kde', color='y', label=f'Iterative ({self.X_train["TotalCharges_itimp"].std():.2f})')
            plt.title("KDE Comparison - Train Data")
            plt.legend()
            plt.subplot(1, 2, 2)
            self.X_test['TotalCharges'].plot(kind='kde', color='r', label=f'Original ({self.X_test["TotalCharges"].std():.2f})')
            self.X_test['TotalCharges_itimp'].plot(kind='kde', color='y', label=f'Iterative ({self.X_test["TotalCharges_itimp"].std():.2f})')
            plt.title("KDE Comparison - Test Data")
            plt.legend()
            plt.tight_layout()
            plt.show()
            self.X_train = self.X_train.drop(['TotalCharges', 'TotalChargesreplaced'], axis=1, errors='ignore')
            self.X_test = self.X_test.drop(['TotalCharges', 'TotalChargesreplaced'], axis=1, errors='ignore')
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    def variable_trans(self):
        try:
            cols = ['MonthlyCharges_itimp', 'TotalCharges_itimp']
            df_sel = self.X_train[cols].copy()
            df_sel1 = self.X_test[cols].copy()
            # Apply all transformations
            # df_sel1 = VariableTransformations.power_transform_check(df_sel, method='yeo-johnson')
            # log.info(df_sel1.isnull().sum())
            log.info(df_sel.columns)
            df_sel2 = VariableTransformations.quantile_transform_check(df_sel)
            log.info(df_sel2.isnull().sum())
            self.X_train['MonthlyCharges_qt'] = df_sel2['MonthlyCharges_itimp_qt']
            self.X_train['TotalCharges_qt'] = df_sel2['TotalCharges_itimp_qt']

            df_sel11 = VariableTransformations.quantile_transform_check(df_sel1)
            log.info(df_sel2.isnull().sum())
            self.X_test['MonthlyCharges_qt'] = df_sel11['MonthlyCharges_itimp_qt']
            self.X_test['TotalCharges_qt'] = df_sel11['TotalCharges_itimp_qt']

            # df_sel3 = VariableTransformations.sqrt_transform_check(df_sel)
            # log.info(df_sel3.isnull().sum())
            # df_sel4 = VariableTransformations.log_transform_check(df_sel)
            # log.info(df_sel4.isnull().sum())
            # df_sel5 = VariableTransformations.exp_transform_check(df_sel)
            # log.info(df_sel5.isnull().sum())
            #
            # df_sel6 = VariableTransformations.cbrt_transform_check(df_sel)
            # log.info(df_sel6.isnull().sum())
            #
            # df_sel7 = VariableTransformations.tanh_transform_check(df_sel)
            # log.info(df_sel7.isnull().sum())
            #
            # df_sel8 = VariableTransformations.quantile_rank_transform_check(df_sel)
            # log.info(df_sel8.isnull().sum())
            #
            # df_sel9 = VariableTransformations.power_transform_check(df_sel, method='box-cox')
            # log.info(df_sel9.isnull().sum())
            #
            # df_sel10 = VariableTransformations.power_trans(df_sel)
            # log.info(df_sel9.isnull().sum())
            #
            # # Add transformed data to main train set
            # self.X_train_num = pd.concat([self.X_train, df_sel1, df_sel2, df_sel3, df_sel4, df_sel5, df_sel6,df_sel7], axis=1)
            # log.info("Variable transformations applied successfully on MonthlyCharges and TotalCharges.")
            #
            # # Visualize each transformation type
            # plot_kde_comparison(df_sel1, '_pt', 'b', 'Power-yeo-johnson')
            # plot_kde_comparison(df_sel2, '_qt', 'g', 'Quantile')
            # plot_kde_comparison(df_sel11, '_qt', 'g', 'Quantile1')
            # plot_kde_comparison(df_sel3, '_sqrt', 'm', 'Sqrt')
            # plot_kde_comparison(df_sel4, '_log', 'c', 'Log')
            # # plot_kde_comparison(df_sel5, '_exp', 'k', 'exp')
            # plot_kde_comparison(df_sel6, '_cbrt', 'y', 'cube')
            # plot_kde_comparison(df_sel7, '_tanh', (0.1, 0.2, 0.5), 'tanh')
            # plot_kde_comparison(df_sel8, '_qtr', (0.1, 0.4, 0.6), 'RankGauss')
            # plot_kde_comparison(df_sel9, '_pt', (0.4, 0.2, 0.9), f'Power-box-cox')
            # plot_kde_comparison(df_sel10, '_tsqrt', (0.4, 0.1, 0.4), f'reciprocal sqrt')
            log.info(f'Check null- {self.X_train.isnull()}\n{self.X_test.isnull().sum()}')

        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')


    def Outlier_main(self):
        try:
            cols = ['MonthlyCharges_qt', 'TotalCharges_qt']
            df1 = self.X_train[cols].copy()
            df2 = self.X_test[cols].copy()
            outlier_final1 = Outlier.Winzos(df1, 'gaussian', 2.5)
            # Outlier.Winzos(df1, 'iqr', 1.5 )
            # Outlier.Winzos(df1, 'mad', 'auto' )
            # Outlier.Winzos(df1, 'quantiles', 0.01)
            # Outlier.Isolation_forests(df1)
            # Outlier.percentail_cap(df1)

            # for test
            outlier_final2 = Outlier.Winzos(df2, 'gaussian', 2.5)
            # Outlier.Winzos(df2, 'iqr', 1.5 )
            # Outlier.Winzos(df2, 'mad', 1.5)
            # Outlier.Winzos(df2, 'quantiles', 0.01)
            # Outlier.Isolation_forests(df2)
            # Outlier.percentail_cap(df)
            log.info(f'X_train_cols:{self.X_train.columns}')
            log.info(f'Outlier columns :{outlier_final1.columns}')

            self.X_train['MonthlyCharges_qt_outl'] = outlier_final1['MonthlyCharges_qt']
            self.X_train['TotalCharges_qt_outl'] = outlier_final1['TotalCharges_qt']

            self.X_test['MonthlyCharges_qt_outl'] = outlier_final2['MonthlyCharges_qt']
            self.X_test['TotalCharges_qt_outl'] = outlier_final2['TotalCharges_qt']

            self.X_train_num = self.X_train.select_dtypes(exclude=['object'])
            self.X_test_num = self.X_test.select_dtypes(exclude=['object'])
            self.X_train_cat = self.X_train.select_dtypes(include=['object'])
            self.X_test_cat = self.X_test.select_dtypes(include=['object'])
            log.info(f'Check the num columns :{self.X_train_num.columns}')
            log.info(f'Check the cat columns :{self.X_train_cat.columns}')
            log.info(self.X_test_cat.shape)
            #Dropping the unnecessary columns 'MonthlyCharges', 'TotalCharges_itimp', 'MonthlyCharges_itimp', 'MonthlyCharges_qt', 'TotalCharges_qt'
            self.X_test_num = self.X_test_num.drop(['MonthlyCharges','TotalCharges_itimp','MonthlyCharges_itimp','MonthlyCharges_qt','TotalCharges_qt'], axis=1)
            self.X_train_num = self.X_train_num.drop(['MonthlyCharges','TotalCharges_itimp','MonthlyCharges_itimp','MonthlyCharges_qt', 'TotalCharges_qt'], axis=1)
            log.info(f'Check the num columns after removing unnecessary columns:{self.X_train_num.columns}')
            # plot_kde_comparison(self.X_train_num,'_qt_outl', 'r', label='check after')
            # log.info(self.y_train.info())
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

    #Now I am starting the feature-selection from below
    def filter_methods(self):
        try:
            nominal_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies',
                            'PaperlessBilling', 'PaymentMethod']

            ordinal_cols = ['Contract']
            log.info(f'Check-label{self.y_train}')
            df1 = Encoding.Ordinal_encoder(self.X_train_cat, ordinal_cols)
            df2 = Encoding.Onehot_encode(self.X_train_cat, nominal_cols)
            df3 = Encoding.LabelEncoding_calumns(self.y_train)
            df4 = Encoding.Ordinal_encoder(self.X_test_cat, ordinal_cols)
            df5 = Encoding.Onehot_encode(self.X_test_cat, nominal_cols)
            df6 = Encoding.LabelEncoding_calumns(self.y_test)


            log.info(df1)
            log.info(df2)
            log.info(df3)
            self.X_train_cat = pd.concat([df1, df2], axis=1)
            # log.info(self.X_train_cat.shape)
            # log.info(self.X_test_cat.shape)
            self.X_test_cat = pd.concat([df4,df5], axis=1)
            # log.info(self.X_test_cat.shape)
            self.X_train_cat['SeniorCitizen'] = self.X_train_num['SeniorCitizen']
            self.X_test_cat['SeniorCitizen'] = self.X_test_num['SeniorCitizen']
            self.X_train_cat['sim'] = self.X_train_cat['sim'].map({'Jio':0, 'Airtel':1, 'Vi':2, 'BSNL':3})
            self.X_test_cat['sim'] = self.X_test_cat['sim'].map({'Jio': 0, 'Airtel': 1, 'Vi': 2, 'BSNL': 3})
            log.info(self.X_test_cat['sim'])
            self.X_train_num = self.X_train_num.drop(['SeniorCitizen'], axis=1)
            self.X_test_num = self.X_test_num.drop(['SeniorCitizen'], axis=1)
            # Now Label encoding saving
            self.y_train = df3
            self.y_test = df6
            log.info(f'Check-label{self.y_train}')
            # checking for object and remove from all dataset
            # cols_object = self.X_train_cat.select_dtypes(include='object').columns
            # log.info(droped_cols)
            # log.info(f'Befor shape was :{self.X_train_cat.shape}')
            # self.X_train_cat = self.X_train_cat.drop(cols_object, axis=1)
            self.X_train_cat1, self.X_test_cat1 = FilterMethod.chisquare(self.X_train_cat, self.X_test_cat, self.y_train)
            log.info(f'Check_new{self.X_train_cat1.shape}')
            log.info(f'Check _new{self.X_test_cat1.shape}')
            self.X_train_cat1['sim'] = self.X_train_cat['sim']
            self.X_test_cat1['sim'] = self.X_test_cat['sim']
            # FilterMethod.chisquare(self.X_train_num, self.y_train) #while runing the it showing negative values can't handle.
            log.info(self.X_train_num.columns)
            # self.X_train_num = FilterMethod.Anova_test(self.X_train_num, self.y_train)
            # print(self.X_train_cat1.shape)
            # print(self.X_test_cat1.shape)
            log.info(self.X_train_cat1)
            log.info(self.X_test_cat1)
            # FilterMethod.multi_classif(self.X_train_cat, self.X_test_cat, self.y_train) # when we doing this method main column like Senior citizen is removing.
            FilterMethod.constant(self.X_train_cat1, self.X_test_cat1) # no changes ratherthan chi2 test
            FilterMethod.quasi_constant(self.X_train_cat1, self.X_test_cat1) # no change ratherthan chi2 test
            log.info(self.X_train_num)
            FilterMethod.constant(self.X_train_num, self.X_test_num)
            FilterMethod.quasi_constant(self.X_train_num, self.X_test_num) # no change came so we are not saving the change and continue for corelation,
            FilterMethod.t_test_filter(self.X_train_num, self.X_test_num, self.y_train)
            FilterMethod.co_relation(self.X_train_num, self.X_test_num, self.y_train)
            log.info(self.y_train)
            log.info(self.y_test)
            log.info(self.X_train_cat.shape)
            log.info(self.X_test_cat.shape)
            log.info(self.X_test_num.shape)
            log.info(self.X_train_num.shape)
            log.info(self.X_test_cat1.columns)
            self.training_data = pd.DataFrame()
            self.testing_data = pd.DataFrame()
            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat1.reset_index(drop=True, inplace=True)
            self.X_train_cat1.reset_index(drop=True, inplace=True)
            #combining the data as self.training_data and self.testing_data
            self.training_data = pd.concat([self.X_train_cat1, self.X_train_num], axis=1)
            self.testing_data = pd.concat([self.X_test_cat1, self.X_test_num], axis=1)
            log.info(f"Training data shape check :{self.training_data.shape}")
            log.info(f"Testing data shape check :{self.testing_data.shape}")
        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')


    def balanced_data(self):
        try:
            log.info('----------------Before Balancing------------------------')
            log.info(f'Total row for Good category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 1)}')
            log.info(f'Total row for Bad category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 0)}')
            log.info(f'---------------After Balancing-------------------------')
            sm = SMOTE(random_state=42)
            self.training_data_res,self.y_train_res = sm.fit_resample(self.training_data,self.y_train)
            log.info(f'Total row for Good category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 1)}')
            log.info(f'Total row for Bad category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 0)}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            log.error(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

        # ✅ Scale only imputed raw MonthlyCharges & TotalCharges
    def scaling_tech(self):
        try:
            backup_year_train = self.training_data_res['YearStayed'].copy()
            backup_year_test = self.testing_data['YearStayed'].copy()

            self.training_data_res.drop(['YearStayed'], axis=1, inplace=True)
            self.testing_data.drop(['YearStayed'], axis=1, inplace=True)

            scale_cols = ['MonthlyCharges_qt_outl', 'TotalCharges_qt_outl']
            self.ms = StandardScaler()
            self.ms.fit(self.training_data_res[scale_cols])

            scaled_train = pd.DataFrame(
                self.ms.transform(self.training_data_res[scale_cols]),
                columns=scale_cols, index=self.training_data_res.index)
            scaled_test = pd.DataFrame(
                self.ms.transform(self.testing_data[scale_cols]),
                columns=scale_cols, index=self.testing_data.index)

            other_train = self.training_data_res.drop(scale_cols, axis=1, errors='ignore')
            other_test = self.testing_data.drop(scale_cols, axis=1, errors='ignore')

            self.training_data_t = pd.concat([other_train, scaled_train], axis=1)
            self.testing_data_t = pd.concat([other_test, scaled_test], axis=1)
            self.training_data_t['YearStayed'] = backup_year_train
            self.testing_data_t['YearStayed'] = backup_year_test

            log.info(self.testing_data_t)
            model = common(self.training_data_t, self.y_train_res, self.testing_data_t, self.y_test)
            log.info("Scaling applied on MonthlyCharges_itimp & TotalCharges_itimp.")

            with open(r"C:\Users\sivan\OneDrive - MSFT\Intership\output\scaler.pkl", "wb") as f:
                pickle.dump(self.ms, f)

            self.training_data_t.to_csv('./Data/final.csv', index=False)
        except Exception:
            et, em, el = sys.exc_info()
            log.error(f"{et} at line {el.tb_lineno}: {em}")

    def sample(self):
        try:
            # self.check_data_ind = self.training_data_t.head(200)
            # self.check_data_dep = self.y_train_res[:200]
            # log.info(self.check_data_dep)
            # che(self.check_data_ind,self.check_data_dep)
            self.model_final = LogisticRegression(C= 100.0, class_weight= None, l1_ratio= None, max_iter= 100, multi_class= 'auto', n_jobs= None, penalty= 'l1', solver= 'liblinear')

            self.model_final.fit(self.training_data_t, self.y_train_res)
            log.info(f'Accuracy of the model: {accuracy_score(self.y_train_res, self.model_final.predict(self.training_data_t))}')
            log.info(f'Test Accuracy : {accuracy_score(self.y_test, self.model_final.predict(self.testing_data_t))}')

            if self.model_final is not None:
                # Save best model
                with open("C:\\Users\\sivan\\OneDrive - MSFT\\Intership\\output\\model.pkl", "wb") as f:
                    pickle.dump(self.model_final, f)

                # # Save scaler
                # with open("C:\\Users\\sivan\\OneDrive - MSFT\\python\\churn_project\\output\\scaler.pkl", "wb") as f:
                #     pickle.dump(self.ms, f)

                log.info("Best Model and Scaler saved successfully as pickle files.")
            else:
                log.error("No model was found")



        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            log.error(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


if __name__ == '__main__':
    obj = Churn()
    obj.handling_missing_values()
    obj.visual_check()
    obj.variable_trans()
    obj.Outlier_main()
    obj.filter_methods()
    obj.balanced_data()
    obj.scaling_tech()
    obj.sample()
