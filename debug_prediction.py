import pandas as pd
import numpy as np
import pickle
import os

def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'models', 'trained_models.pkl'), 'rb') as f:
        models = pickle.load(f)
    with open(os.path.join(base_dir, 'models', 'preprocessors.pkl'), 'rb') as f:
        pre = pickle.load(f)
    return models, pre

def process_inference(payload, pre):
    df = pd.DataFrame([payload])
    df['policy_bind_date'] = pd.to_datetime(df.get('policy_bind_date', '2010-01-01'))
    df['incident_date'] = pd.to_datetime(df.get('incident_date', '2015-01-10'))
    df['policy_tenure'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    df['incident_month'] = df['incident_date'].dt.month
    df['incident_season'] = df['incident_month'].apply(lambda x: (x%12 // 3) + 1)
    
    # Severity Map: {'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4}
    df['incident_severity'] = df['incident_severity'].map(pre['severity_map']).fillna(2)
    
    # Pre-select columns to drop from payload that aren't in training
    cols_to_drop = ['policy_bind_date', 'incident_date', 'policy_number', 'insured_zip', 'incident_location', 'insured_hobbies']
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_drops, inplace=True)
    
    df_encoded = pd.get_dummies(df)
    
    final_df = pd.DataFrame(np.zeros((1, len(pre['columns']))), columns=pre['columns'], dtype=np.float64)
    for col in pre['columns']:
        if col in df_encoded.columns:
            final_df.loc[0, col] = float(df_encoded.loc[0, col])
            
    final_df = final_df.astype(np.float64)
    scaled = pre['scaler'].transform(final_df)
    pca_data = pre['pca'].transform(scaled)
    return pca_data

# FRAUD CASE (From Dataset row 1: PhD, Major Damage, NO police report, 63400 claim)
fraud_payload = {
    "age": 48,
    "insured_sex": "FEMALE",
    "insured_education_level": "PhD",
    "insured_occupation": "adm-clerical",
    "insured_relationship": "other-relative",
    "policy_state": "OH",
    "policy_csl": "250/500",
    "policy_deductable": 1000,
    "policy_annual_premium": 1415,
    "umbrella_limit": 0,
    "incident_severity": "Major Damage",
    "total_claim_amount": 71610,
    "incident_type": "Single Vehicle Collision",
    "collision_type": "Side Collision",
    "number_of_vehicles_involved": 1,
    "property_damage": "YES",
    "police_report_available": "YES"
}

models, pre = load_resources()
X = process_inference(fraud_payload, pre)

print("--- RAW PREDICTIONS ---")
for name, model in models.items():
    prob = model.predict_proba(X)[0][1]
    print(f"{name}: {prob:.1%}")

avg_prob = (models['Random Forest'].predict_proba(X)[0][1] * 0.6) + (models['SVM (RBF)'].predict_proba(X)[0][1] * 0.4)
print(f"Final Ensemble Probability: {avg_prob:.1%}")
