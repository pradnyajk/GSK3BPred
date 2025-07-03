# -*- coding: utf-8 -*-
"""
Created on Sat May 10 17:34:31 2025

@author: Wisdom
"""

"""
gsk3bpred: Predict molecular activity using a pre-trained DNN model with Mordred descriptors.
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disables oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR


# Function to calculate Mordred descriptors
def calculate_mordred_descriptors(smiles_list):
    calc = Calculator(descriptors, ignore_3D=False)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    df = calc.pandas(mols)
    return df

def main():
    print("Loading SMILES data...")
    data_smiles = pd.read_csv(sys.argv[1])
    smiles = data_smiles['SMILES']

    print("Calculating Mordred descriptors...")
    mordred_descriptors = calculate_mordred_descriptors(smiles)

    print("Loading training data for reference features...")
    train_data = pd.read_csv('/GSK3BPred/X_train.csv')

    # Keep only common columns
    common_columns = [col for col in mordred_descriptors.columns if col in train_data.columns]
    filtered_descriptors = mordred_descriptors[common_columns]

    print("Cleaning descriptors...")
    non_numeric_columns = filtered_descriptors.select_dtypes(exclude=['number']).columns
    cleaned_descriptors = filtered_descriptors.copy()

    for col in non_numeric_columns:
        train_numeric = pd.to_numeric(train_data[col], errors='coerce')
        mean_value = train_numeric.mean()
        cleaned_descriptors[col] = pd.to_numeric(cleaned_descriptors[col], errors='coerce')
        cleaned_descriptors[col] = cleaned_descriptors[col].fillna(mean_value)

    cleaned_descriptors = cleaned_descriptors.apply(pd.to_numeric, errors='coerce')

    # Ensure column order matches training data
    print("Normalizing descriptors...")
    cleaned_descriptors_ordered = cleaned_descriptors[train_data.columns]

    scaler = joblib.load('/GSK3BPred/mordred_scaler.pkl')
    cleaned_descriptors_scaled = pd.DataFrame(
        scaler.transform(cleaned_descriptors_ordered),
        columns=train_data.columns
    )

    print("Loading DNN model and making predictions...")
    model = load_model('/GSK3BPred/mordred_dnn_model.h5')
    predictions = model.predict(cleaned_descriptors_scaled)
    
    # Handle binary output from sigmoid activation
    probabilities = predictions.flatten()
    predicted_labels = ['Inhibitor' if prob >= 0.5 else 'Non-inhibitor' for prob in probabilities]
    confidence = [ prob if prob >= 0.5 else 1 - prob for prob in probabilities]
    print("Saving prediction results...")
    result_df = pd.DataFrame({
        'SMILES': smiles,
        'Predicted Class': predicted_labels,
        'Probability of Inhibitor': probabilities,
        'Confidence of Predicted Class': confidence
    })

    result_df.to_csv('Prediction_Results.csv', index=False)
    print("Prediction completed. Results saved to 'Prediction_Results.csv'.")

if __name__ == '__main__':
    main()
