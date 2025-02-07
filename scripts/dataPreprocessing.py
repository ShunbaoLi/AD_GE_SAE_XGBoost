import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def preprocess_data(adni_path, anm1_path, anm2_path, output_path='preprocessed_gene_expression.csv'):
    adni_data = pd.read_csv(adni_path, index_col=0)
    anm1_data = pd.read_csv(anm1_path, index_col=0)
    anm2_data = pd.read_csv(anm2_path, index_col=0)

    common_probes = list(set(adni_data.columns) & set(anm1_data.columns) & set(anm2_data.columns))
    adni_data = adni_data[common_probes]
    anm1_data = anm1_data[common_probes]
    anm2_data = anm2_data[common_probes]

    data_combined = pd.concat([adni_data, anm1_data, anm2_data], axis=0)

    scaler = RobustScaler()
    data_normalised = pd.DataFrame(scaler.fit_transform(data_combined), 
                                   index=data_combined.index, 
                                   columns=data_combined.columns)

    # Save preprocessed data
    data_normalised.to_csv(output_path)
    return data_normalised