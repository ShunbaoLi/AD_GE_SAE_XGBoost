import pandas as pd
import torch
from scripts.data_preprocessing import preprocess_data
from scripts.autoencoder import SparseAutoencoder, train_autoencoder
from scripts.xgboost_classifier import XGBoostClassifier
from scripts.fast_recursive_feature_elimination import fast_recursive_feature_selection

def main():
    preprocessed_data = preprocess_data('adni_gene_expression.csv', 
                                        'anm1_gene_expression.csv', 
                                        'anm2_gene_expression.csv')
    X = preprocessed_data.values
    
    labels = pd.read_csv('labels.csv').values.squeeze() 

    input_dim = X.shape[1]
    hidden_dim = 100  # Example hidden layer size
    model = SparseAutoencoder(input_dim, hidden_dim)
    X_tensor = torch.from_numpy(X).float()
    trained_model = train_autoencoder(model, X_tensor)

    # Extract features
    with torch.no_grad():
        features, _ = trained_model(X_tensor)

    # Classification with XGBoost
    xgb_classifier = XGBoostClassifier()
    xgb_classifier.train(features.numpy(), labels)
    accuracy, report = xgb_classifier.evaluate(features.numpy(), labels)
    print(f'Accuracy: {accuracy}\n{report}')

    # Feature Importance and Selection
    importance = xgb_classifier.feature_importance()
    optimal_features = fast_recursive_feature_selection(features.numpy(), labels)
    print(f'Optimal features: {optimal_features}')

if __name__ == "__main__":
    main()