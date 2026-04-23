import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

def clean_data(file_path):
    # Get absolute path to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'insurance_claim.csv')
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Cleaning: Handle missing values '?'
    # As per Module 3, we'll use mode for categorical and median for numerical (if any)
    # Checking which columns have '?'
    cols_with_missing = [col for col in df.columns if col in df.columns and df[col].dtype == 'object' and '?' in df[col].unique()]
    for col in cols_with_missing:
        mode_val = df[df[col] != '?'][col].mode()[0]
        df[col] = df[col].replace('?', mode_val)
        
    # Filter to only keep features used in the UI
    ui_features = [
        'age', 'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_relationship',
        'policy_state', 'policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'policy_csl',
        'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted',
        'number_of_vehicles_involved', 'property_damage', 'police_report_available', 'total_claim_amount',
        'fraud_reported', 'policy_bind_date', 'incident_date'
    ]
    df = df[ui_features]
    
    # Feature Engineering (Phase 1)
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    
    # Create policy_tenure (days)
    df['policy_tenure'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    
    # Extract seasonal/monthly features
    df['incident_month'] = df['incident_date'].dt.month
    df['incident_season'] = df['incident_month'].apply(lambda x: (x%12 // 3) + 1)
    
    # Drop dates and non-predictive IDs
    cols_to_drop = ['policy_bind_date', 'incident_date']
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Encoding
    # Ordinal: incident_severity
    severity_map = {'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4}
    df['incident_severity'] = df['incident_severity'].map(severity_map)
    
    # Target Encoding
    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})
    
    # Categorical Columns for One-Hot Encoding
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Split features and target
    X = df.drop('fraud_reported', axis=1)
    y = df['fraud_reported']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA (Module 2)
    # We'll keep components that explain 95% of variance
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    # SMOTE (Handling Imbalance)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_pca, y)
    
    # Save the cleaned data and preprocessors
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'preprocessors.pkl'), 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'pca': pca,
            'columns': X.columns,
            'severity_map': severity_map
        }, f)
        
    # Save processed data for training
    processed_df = pd.DataFrame(X_res)
    processed_df['target'] = y_res
    output_path = os.path.join(base_dir, 'data', 'cleaned_insurance_data.csv')
    processed_df.to_csv(output_path, index=False)
    
    print(f"Data cleaning complete. Cleaned data shape: {processed_df.shape}")
    print(f"PCA components: {pca.n_components_}")

if __name__ == "__main__":
    clean_data('')
