# Vehicle Insurance Fraud Detection SOC 🛡️

A world-class, premium executive dashboard for detecting automobile insurance fraud using ensemble machine learning and forensic data analytics.

## 🚀 Key Features
- **Heads-Up Display (HUD)**: Real-time fraud probability scores with dynamic risk coloring.
- **Ensemble Intelligence**: Combines Random Forest and SVM (RBF) for high-accuracy forensic analysis.
- **Radar Risk Profiling**: Visualizes multi-axis anomalies across Financial, Behavioral, and Documentary dimensions.
- **Actionable Triage**: Persistent sidebar for tracking high-risk claims requiring immediate investigator attention.
- **Ultra-Minimalist UI**: Specialized "Light Executive" industrial design for professional clarity.

## 📦 Project Structure
- `app.py`: Main Streamlit portal and UI engine.
- `preprocessing/`: Core logic for data cleaning, model training, and feature metadata.
- `models/`: High-performance serialized model files (.pkl).
- `data/`: House for the `insurance_claim.csv` dataset.
- `requirements.txt`: Project dependencies and libraries.

## 🛠️ Setup & Usage
1. **Clone the repository**:
   ```bash
   git clone [your-repo-link]
   cd insurance-fraud-system
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Portal**:
   ```bash
   streamlit run app.py
   ```

## 🧠 Model Strategy
The system uses focused 42-feature subsets to maintain 100% parity between training and inference data. It handles class imbalance with `balanced` weight strategies to ensure high-risk claims are correctly identified as **"NOT SECURED."**

---
*Developed for Executive Fraud Intelligence.*
