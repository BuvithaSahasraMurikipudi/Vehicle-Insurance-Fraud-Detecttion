import pickle
import pandas as pd
import numpy as np
import os

def verify():
    base_dir = '.'
    models_dir = os.path.join(base_dir, 'models')
    pre_path = os.path.join(models_dir, 'preprocessors.pkl')
    models_path = os.path.join(models_dir, 'trained_models.pkl')
    
    with open(pre_path, 'rb') as f:
        pre = pickle.load(f)
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
        
    # Sample input data
    input_data = {
        'policy_annual_premium': 1406.91,
        'policy_deductable': 1000,
        'policy_state': 'OH',
        'policy_csl': '250/500',
        'umbrella_limit': 0,
        'policy_bind_date': '2014-10-17',
        'policy_number': 521585,
        'incident_severity': 'Major Damage',
        'total_claim_amount': 71610,
        'incident_date': '2015-01-25',
        'incident_type': 'Single Vehicle Collision',
        'collision_type': 'Side Collision',
        'number_of_vehicles_involved': 1,
        'property_damage': 'YES',
        'police_report_available': 'YES',
        'age': 48,
        'months_as_customer': 328,
        'insured_sex': 'MALE',
        'insured_education_level': 'MD',
        'insured_occupation': 'craft-repair',
        'insured_relationship': 'husband',
        'insured_zip': 466132,
        'incident_location': '9935 4th Drive'
    }
    
    df_raw = pd.DataFrame([input_data])
    
    # Preprocessing
    df_raw['policy_bind_date'] = pd.to_datetime(df_raw['policy_bind_date'])
    df_raw['incident_date'] = pd.to_datetime(df_raw['incident_date'])
    df_raw['policy_tenure'] = (df_raw['incident_date'] - df_raw['policy_bind_date']).dt.days
    df_raw['incident_month'] = df_raw['incident_date'].dt.month
    df_raw['incident_season'] = df_raw['incident_month'].apply(lambda x: (x%12 // 3) + 1)
    df_raw['incident_severity'] = df_raw['incident_severity'].map(pre['severity_map'])
    
    df_processed = pd.get_dummies(df_raw)
    cols_to_drop = ['policy_number', 'policy_bind_date', 'incident_date', 'incident_location', 'insured_zip']
    df_processed.drop(columns=[c for c in cols_to_drop if c in df_processed.columns], inplace=True)
    
    final_df = pd.DataFrame(columns=pre['columns'])
    for col in pre['columns']:
        if col in df_processed.columns:
            final_df.loc[0, col] = df_processed.loc[0, col]
        else:
            final_df.loc[0, col] = 0
            
    X_scaled = pre['scaler'].transform(final_df)
    X_pca = pre['pca'].transform(X_scaled)
    
    print(f"Verification Success!")
    print(f"Original columns: {len(pre['columns'])}")
    print(f"PCA components: {X_pca.shape[1]}")
    
    # Simple test prediction
    for name, model in models.items():
        pred = model.predict(X_pca)[0]
        prob = model.predict_proba(X_pca)[0][1]
        print(f"{name}: Prediction={pred}, Prob={prob:.4f}")

if __name__ == "__main__":
    verify()
