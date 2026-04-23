import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

def generate_importance():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # We need the data JUST before PCA
    # I'll re-run a snippet of the cleaning logic to get the dummy-encoded X
    data_path = os.path.join(base_dir, 'data', 'insurance_claim.csv')
    df = pd.read_csv(data_path)
    
    # Minimal cleaning for importance
    cols_with_missing = [col for col in df.columns if col in df.columns and df[col].dtype == 'object' and '?' in df[col].unique()]
    for col in cols_with_missing:
        mode_val = df[df[col] != '?'][col].mode()[0]
        df[col] = df[col].replace('?', mode_val)
        
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    df['policy_tenure'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    
    severity_map = {'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4}
    df['incident_severity'] = df['incident_severity'].map(severity_map)
    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})
    
    cols_to_drop = ['policy_number', 'policy_bind_date', 'incident_date', 'incident_location', 'insured_zip']
    df.drop(columns=cols_to_drop, inplace=True)
    
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    X = df.drop('fraud_reported', axis=1)
    y = df['fraud_reported']
    
    # Train RF on ORIGINAL features (no PCA)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get importance
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # Save for App
    models_dir = os.path.join(base_dir, 'models')
    with open(os.path.join(models_dir, 'feature_importance.pkl'), 'wb') as f:
        pickle.dump({
            'importances': importances,
            'top_features': importances.head(20).index.tolist(),
            'rf_model_interpret': rf
        }, f)
        
    print(f"Feature importance saved. Top feature: {importances.index[0]}")

if __name__ == "__main__":
    generate_importance()
