import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def fast_recursive_feature_selection(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    original_acc = accuracy_score(y_test, model.predict(X_test))
    
    feature_importances = model.get_booster().get_score(importance_type='gain')
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    
    gains = np.array([importance for _, importance in sorted_features])
    features = np.array([int(feature[1:]) for feature, _ in sorted_features])  # Convert feature names to indices
    
    mu = np.mean(gains)
    sigma = np.std(gains)
    delta = mu
    
    a, b = 0, float('inf')
    
    while True:
        selected_features = features[gains > delta]
        model.fit(X_train[:, selected_features], y_train)
        acc = accuracy_score(y_test, model.predict(X_test[:, selected_features]))
        
        if abs(acc - original_acc) < 0.01:
            a = delta
            delta += sigma
        else:
            b = delta
            break
    
    while a < b:
        delta = (a + b) / 2
        selected_features = features[gains > delta]
        model.fit(X_train[:, selected_features], y_train)
        acc = accuracy_score(y_test, model.predict(X_test[:, selected_features]))
        
        if abs(acc - original_acc) < 0.01:
            a = delta
        else:
            b = delta
        
        if len(selected_features) == 1:
            break
    
    optimal_features = features[gains > a]
    return optimal_features