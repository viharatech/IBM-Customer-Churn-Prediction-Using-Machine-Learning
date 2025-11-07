from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from logger import Logger
log = Logger.get_logs('check_best')
import sys

def  che(train_ind,train_dep):
    try:
        log.info("Started....")
        reg = LogisticRegression()
        reg.fit(train_ind,train_dep)
        log.info(f'test accuracy : {accuracy_score(train_dep, reg.predict(train_ind))}')

        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
            'class_weight': [None, 'balanced'],
            'max_iter': [100, 200, 500, 1000],
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'n_jobs': [None, -1],
            'l1_ratio': [None, 0.0, 0.25, 0.5, 0.75, 1.0]  # only used if penalty='elasticnet'
        }

        grid = GridSearchCV(reg, param_grid, cv=5, scoring='accuracy')
        grid.fit(train_ind, train_dep)
        log.info(f'Best :{grid.best_params_}')
        log.info(f'Best score:{grid.best_score_}')
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        log.error(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')
