import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_insurance_data.csv')
    models_path = os.path.join(base_dir, 'models', 'trained_models.pkl')
    eval_dir = os.path.join(base_dir, 'evaluations')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # We'll use the same split as train_models.py (standardize this in practice)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load models
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
        
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(eval_dir, f'cm_{name.lower().replace(" ", "_")}.png'))
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(0) # Plot all on one ROC chart
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
        # Print Classification Report
        print(f"\n--- {name} Evaluation ---")
        print(classification_report(y_test, y_pred))

    plt.figure(0)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(eval_dir, 'roc_curves.png'))
    plt.close()
    
    print("\nEvaluation complete. Charts saved to 'evaluations/' directory.")

if __name__ == "__main__":
    evaluate()
