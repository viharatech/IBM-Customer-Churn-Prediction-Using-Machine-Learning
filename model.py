import sys

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import xgboost as xgb
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC
from logger import Logger
import pickle
logger = Logger.get_logs('model')

def common(X_train, y_train, X_test, y_test):
    """Train, evaluate, select best by AUC, and return best model"""
    try:
        classifiers = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(criterion='entropy'),
            "Random Forest": RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=42),
            "Xgboost": XGBClassifier(),
            "SVM":SVC(kernel='rbf'),
            'Gradient':GradientBoostingClassifier()
        }

        plt.figure(figsize=(8, 6))

        best_auc = 0
        best_model = None
        best_name = ""

        for name, model in classifiers.items():
            model.fit(X_train, y_train)
            with open(f'C:\\Users\\sivan\\OneDrive - MSFT\\Intership\\output\\{name}.pkl', 'wb') as f:
                pickle.dump(model, f)
            y_pred = model.predict(X_test)

            # Log metrics
            logger.info(f"------ {name} ------")
            logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
            logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            # Get probability scores for ROC
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except:
                y_prob = model.decision_function(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            logger.info(f"AUC Score ({name}): {roc_auc:.4f}")
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

            # Track best model by AUC
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_model = model
                best_name = name

        # Random guess line
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - All Models")
        plt.legend(loc="lower right")
        plt.show()

        logger.info(f"Best model selected: {best_name} with AUC={best_auc:.4f}")
        return best_model

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue at {er_lin.tb_lineno} : {er_msg}')
        return None
