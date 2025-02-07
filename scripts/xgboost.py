import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class XGBoostClassifier:
    def __init__(self, params=None):
        if params is None:
            params = {
                'objective': 'multi:softmax',
                'num_class': 3,  # Assuming three classes: NC, MCI, AD
                'eval_metric': 'mlogloss',
                'eta': 0.1,
                'max_depth': 6,
                'lambda': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
        self.params = params
        self.model = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=100):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'eval'))
        
        self.model = xgb.train(self.params, dtrain, num_boost_round, evals, early_stopping_rounds=10, verbose_eval=True)
    
    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report
    
    def feature_importance(self):
        importance = self.model.get_score(importance_type='gain')
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)