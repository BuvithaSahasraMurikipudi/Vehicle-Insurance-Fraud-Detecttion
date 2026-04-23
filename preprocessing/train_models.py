import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_models():
    # Get absolute path to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_insurance_data.csv')
    
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models with class balance handling
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced_subsample'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    }
    
    results = {}
    trained_models = {}
    
    print("Starting Model Training and Comparison...")
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Fit on whole train set
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'CV Mean': cv_scores.mean(),
            'Test Accuracy': acc,
            'Classification Report': classification_report(y_test, y_pred, output_dict=True)
        }
        trained_models[name] = model
        
        print(f"{name}: CV={cv_scores.mean():.4f}, Test Acc={acc:.4f}")
        
    # Phase 3: Detailed Evaluation
    # Save the results and models
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, 'trained_models.pkl'), 'wb') as f:
        pickle.dump(trained_models, f)
        
    # Save results summary for Phase 3 documentation
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'CV Mean': [r['CV Mean'] for r in results.values()],
        'Test Acc': [r['Test Accuracy'] for r in results.values()]
    })
    results_df.to_csv(os.path.join(base_dir, 'data', 'model_performance.csv'), index=False)
    
    print("\nModel training complete. Results saved.")

if __name__ == "__main__":
    train_models()
