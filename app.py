import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="FinRisk | Executive Analytics", layout="wide", page_icon="🛡️")

# --- Initialize Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Executive Theme (Hardcoded Obsidian Dark, Large Font) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    :root {
        --bg-color: #ffffff;
        --sidebar-bg: #f8f9fa;
        --text-main: #000000;
        --text-muted: #4b5563;
        --accent-primary: #2563eb;
        --accent-warning: #d97706;
        --accent-critical: #dc2626;
        --border-color: #e5e7eb;
    }

    /* Global Typography & Core Reset */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: var(--bg-color) !important;
        color: var(--text-main) !important;
        font-size: 19px !important;
        line-height: 1.6;
    }

    * { color: var(--text-main) !important; } /* Force Black Visibility */

    /* Hide Streamlit Clutter */
    [data-testid="stHeader"], [data-testid="stToolbar"] { visibility: hidden !important; height: 0; }

    .block-container { padding-top: 1rem !important; }

    /* Flat Content Sections (No Boxes) */
    .glass-card {
        background: transparent;
        border: none !important;
        padding: 0;
        margin-bottom: 2rem;
        box-shadow: none !important;
    }
    
    .score-hud {
        text-align: center;
        background: #f8f9fa;
        padding: 2.5rem;
        border-radius: 20px;
        border: none !important;
    }

    .dashboard-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #000000 !important;
        margin-bottom: 2rem !important;
        text-transform: uppercase;
    }

    /* Uniform Input Parameters Strip Style */
    input, select, textarea, div[data-baseweb="select"] *, [data-baseweb="tab-list"] button {
        background-color: #f9fafb !important;
        color: #000000 !important;
        border: none !important;
        border-bottom: 2px solid #e5e7eb !important;
        border-radius: 0 !important;
        font-size: 1.1rem !important;
    }
    
    input:focus, div[data-baseweb="select"] *:focus {
        border-bottom-color: var(--accent-primary) !important;
    }
    
    label { 
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        color: #374151 !important; 
        margin-bottom: 8px !important; 
    }

    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] * { color: #000000 !important; }

    /* Custom Tooltip/Info badges */
    .risk-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        border: 1px solid transparent;
    }
    .badge-high { background: rgba(255, 75, 75, 0.1); color: var(--accent-critical); border-color: var(--accent-critical); }
    .badge-low { background: rgba(0, 212, 255, 0.1); color: var(--accent-primary); border-color: var(--accent-primary); }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent !important; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent-primary) !important; }
</style>
""", unsafe_allow_html=True)

# --- Resource Loading ---
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(base_dir, 'models', 'trained_models.pkl')
    pre_path = os.path.join(base_dir, 'models', 'preprocessors.pkl')
    imp_path = os.path.join(base_dir, 'models', 'feature_importance.pkl')
    
    with open(models_path, 'rb') as f: models = pickle.load(f)
    with open(pre_path, 'rb') as f: pre = pickle.load(f)
    with open(imp_path, 'rb') as f: imp = pickle.load(f)
    
    options = {
        "insured_occupation": ["craft-repair", "machine-op-inspct", "sales", "armed-forces", "tech-support", "prof-specialty", "other-service", "priv-house-serv", "exec-managerial", "protective-serv", "transport-moving", "handlers-cleaners", "adm-clerical", "farming-fishing"],
        "insured_education_level": ["MD", "PhD", "Associate", "Masters", "High School", "College", "JD"],
        "incident_severity": ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"],
        "policy_state": ["OH", "IN", "IL"],
        "policy_csl": ["250/500", "100/300", "500/1000"],
        "incident_type": ["Single Vehicle Collision", "Vehicle Theft", "Multi-vehicle Collision", "Parked Car"],
        "collision_type": ["Side Collision", "Rear Collision", "Front Collision", "None"],
        "property_damage": ["YES", "NO"],
        "police_report_available": ["YES", "NO"],
        "authorities_contacted": ["Police", "Fire", "Other", "Ambulance", "None"],
        "insured_sex": ["MALE", "FEMALE", "OTHER"],
        "insured_relationship": ["husband", "other-relative", "own-child", "unmarried", "wife", "not-in-family"]
    }
    for k in ["insured_occupation", "insured_education_level", "incident_type", "collision_type", "policy_state", "policy_csl"]:
        options[k] = ["OTHERS"] + options[k]
        
    return models, pre, imp, options

def process_inference(payload, pre):
    df = pd.DataFrame([payload])
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    df['policy_tenure'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    df['incident_month'] = df['incident_date'].dt.month
    df['incident_season'] = df['incident_month'].apply(lambda x: (x%12 // 3) + 1)
    df['incident_severity'] = df['incident_severity'].map(pre['severity_map']).fillna(2)
    df_encoded = pd.get_dummies(df)
    final_df = pd.DataFrame(np.zeros((1, len(pre['columns']))), columns=pre['columns'], dtype=np.float64)
    for col in pre['columns']:
        if col in df_encoded.columns:
            final_df.loc[0, col] = float(df_encoded.loc[0, col])
    final_df = final_df.astype(np.float64)
    scaled = pre['scaler'].transform(final_df)
    pca_data = pre['pca'].transform(scaled)
    return pca_data

def draw_radar(payload, score):
    categories = ['Financial Intensity', 'Behavioral Anomaly', 'Documentary Risk', 'Policy Tenure', 'Exposure Radius']
    
    # Heuristic scoring for demo-radar visibility
    val_fin = min(1.0, payload.get('total_claim_amount', 0) / 100000)
    val_beh = 0.8 if payload.get('incident_severity') == 'Major Damage' else 0.4
    val_doc = 0.2 if payload.get('police_report_available') == 'YES' else 0.9
    val_pol = min(1.0, payload.get('age', 35) / 60)
    val_exp = score
    
    values = [val_fin, val_beh, val_doc, val_pol, val_exp]
    values += values[:1] # close the loop
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('none')
    ax.set_facecolor((0, 0, 0, 0.02)) # Fix: Use tuple for RGBA
    
    ax.fill(angles, values, color='#3b82f6', alpha=0.3)
    ax.plot(angles, values, color='#3b82f6', linewidth=2)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='#1a1b1e', size=8) # Darker text
    
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(color=(0, 0, 0, 0.05)) # Fix: Use tuple
    
    return fig

def main():
    models, pre, imp, opts = load_resources()

    with st.sidebar:
        st.markdown("<div class='sidebar-logo'>FINRISK SOC</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Command Node</div>", unsafe_allow_html=True)
        menu = st.radio("Access Level", ["🔒 Forensic Assessment", "📈 Intelligence Dashboard", "📂 Case Ledger"], index=0, label_visibility="collapsed")
        
        st.markdown("<div class='section-label' style='margin-top:40px;'>High-Risk Triage</div>", unsafe_allow_html=True)
        high_risk_cases = [c for c in st.session_state.history if c['score'] > 0.6]
        if not high_risk_cases:
            st.markdown("<div style='font-size:0.8rem; color:var(--text-muted); opacity:0.5;'>No active threats detected.</div>", unsafe_allow_html=True)
        for case in high_risk_cases[:5]:
            st.markdown(f"<div class='triage-card triage-critical'><b>CaseID: #{case['id']}</b><br>Score: {case['score']:.1%}<br><small style='opacity:0.6;'>{case['time']}</small></div>", unsafe_allow_html=True)

    if menu == "🔒 Forensic Assessment":
        st.markdown("<div class='dashboard-title'>Claim Forensic analyzer</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:var(--text-muted); margin-bottom:2rem;'>Executive Interface for Autonomous Fraud Detection.</p>", unsafe_allow_html=True)

        col_inp, col_res = st.columns([2, 1.2])

        payload = {}
        with col_inp:
            t1, t2, t3 = st.tabs(["Insured Entity", "Policy Coverage", "Loss Event"])
            with t1:
                c1, c2 = st.columns(2)
                with c1:
                    payload["age"] = st.number_input("Entity Age", 18, 100, 35)
                    payload["insured_sex"] = st.selectbox("Entity Gender", opts["insured_sex"])
                with c2:
                    s_edu = st.selectbox("Education Profile", opts["insured_education_level"])
                    payload["insured_education_level"] = st.text_input("Manual Profile", "Advanced") if s_edu == "OTHERS" else s_edu
                    s_occ = st.selectbox("Professional Sector", opts["insured_occupation"])
                    payload["insured_occupation"] = st.text_input("Manual Sector", "Tech") if s_occ == "OTHERS" else s_occ
                payload["insured_relationship"] = st.selectbox("Relationship Status", opts["insured_relationship"])

            with t2:
                c1, c2 = st.columns(2)
                with c1:
                    payload["policy_annual_premium"] = st.number_input("Annual Premium ($)", 0.0, 10000.0, 1100.0)
                    payload["policy_deductable"] = st.number_input("Policy Deductible ($)", 0, 5000, 1000)
                with c2:
                    s_csl = st.selectbox("Liability (CSL)", opts["policy_csl"])
                    payload["policy_csl"] = st.text_input("Manual CSL", "250/500") if s_csl == "OTHERS" else s_csl
                    s_stat = st.selectbox("Policy State", opts["policy_state"])
                    payload["policy_state"] = st.text_input("Manual State", "OH") if s_stat == "OTHERS" else s_stat
                payload["policy_bind_date"] = st.date_input("Underwriting Date")
                payload["umbrella_limit"] = st.number_input("Umbrella Provision ($)", 0, 10000000, 0)

            with t3:
                c1, c2 = st.columns(2)
                with c1:
                    payload["incident_severity"] = st.selectbox("Incident Severity", opts["incident_severity"])
                    payload["total_claim_amount"] = st.number_input("Total Claim Amount ($)", 0, 200000, 15000)
                with c2:
                    payload["incident_date"] = st.date_input("Event Timestamp")
                    payload["authorities_contacted"] = st.selectbox("Authorities Contacted", opts["authorities_contacted"])
                    s_itype = st.selectbox("Incident Type", opts["incident_type"])
                    payload["incident_type"] = st.text_input("Manual Class", "Collision") if s_itype == "OTHERS" else s_itype
                s_ctype = st.selectbox("Collision Mechanism", opts["collision_type"])
                payload["collision_type"] = st.text_input("Manual Mechanism", "Rear") if s_ctype == "OTHERS" else s_ctype
                payload["number_of_vehicles_involved"] = st.number_input("Involved Entities", 1, 10, 1)
                c_p1, c_p2 = st.columns(2)
                with c_p1: payload["property_damage"] = st.selectbox("Asset Damage", opts["property_damage"])
                with c_p2: payload["police_report_available"] = st.selectbox("Police Report Available", opts["police_report_available"])

            payload["policy_number"] = 0
            payload["insured_zip"] = 0
            payload["incident_location"] = "unknown"
            payload["insured_hobbies"] = "reading" 

            if st.button("EXECUTE FORENSIC SCORE ENGINE"):
                with st.spinner("Analyzing Intelligence Layers..."):
                    try:
                        X_pca = process_inference(payload, pre)
                        rf_prob = models['Random Forest'].predict_proba(X_pca)[0][1]
                        svm_prob = models['SVM (RBF)'].predict_proba(X_pca)[0][1]
                        # Boost SVM weight as it is more sensitive in our debug tests
                        avg_prob = (rf_prob * 0.4) + (svm_prob * 0.6) 
                        
                        case_id = f"{len(st.session_state.history) + 1024}"
                        current_time = datetime.now().strftime("%H:%M:%S")
                        st.session_state.history.append({
                            "id": case_id, "score": avg_prob, "time": current_time, 
                            "type": payload['incident_type'], "amount": payload['total_claim_amount']
                        })
                        st.session_state.current_result = {"score": avg_prob, "payload": payload, "rf": rf_prob, "svm": svm_prob}
                        
                    except Exception as e:
                        st.error(f"Inference Logic Fault: {str(e)}")

        with col_res:
            if "current_result" in st.session_state:
                res = st.session_state.current_result
                score = res['score']
                st.markdown("<div class='score-hud'>", unsafe_allow_html=True)
                st.markdown("<div class='hud-label'>ANOMALY COEFFICIENT</div>", unsafe_allow_html=True)
                
                # Dynamic HUD Color
                hud_color = "var(--accent-critical)" if score > 0.5 else ("var(--accent-warning)" if score > 0.3 else "var(--accent-primary)")
                st.markdown(f"<div class='hud-value' style='color:{hud_color};'>{score:.1%}</div>", unsafe_allow_html=True)
                
                if score > 0.5: st.markdown("<div class='risk-badge badge-high'>🚨 CRITICAL ALERT / NOT SECURED</div>", unsafe_allow_html=True)
                elif score > 0.3: st.markdown("<div class='risk-badge' style='background:rgba(255,183,3,0.1); color:var(--accent-warning); border-color:var(--accent-warning);'>⚠️ MODERATE RISK</div>", unsafe_allow_html=True)
                else: st.markdown("<div class='risk-badge badge-low'>✅ SECURE / VERIFIED</div>", unsafe_allow_html=True)
                
                st.markdown("<div style='margin-top:20px; opacity:0.8;'>", unsafe_allow_html=True)
                st.progress(score)
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Key Indicators Card
                st.markdown("<div class='glass-card' style='margin-top:2rem;'>", unsafe_allow_html=True)
                st.markdown("<div class='section-label'>Strategic Indicators</div>", unsafe_allow_html=True)
                indicators = []
                if res['payload']['incident_severity'] in ['Major Damage', 'Total Loss']:
                    indicators.append("Impact profile suggests significant liability risk.")
                if res['payload']['total_claim_amount'] > 50000:
                    indicators.append("Claim value exceeds historical normalcy bounds.")
                for ind in indicators[:3]:
                    st.markdown(f"<div style='border-left:3px solid {hud_color}; padding-left:10px; margin-bottom:10px; font-size:0.9rem;'>{ind}</div>", unsafe_allow_html=True)
                if not indicators:
                    top_feats = imp['importances'].head(2).index
                    for f in top_feats:
                        st.markdown(f"<div style='font-size:0.9rem;'>Driver: {f.replace('_', ' ').title()}</div>", unsafe_allow_html=True)
                # Radar Analysis
                st.markdown("<div class='section-label' style='margin-top:2rem;'>Multi-Axis Risk Profile</div>", unsafe_allow_html=True)
                radar_fig = draw_radar(res['payload'], score)
                st.pyplot(radar_fig)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='glass-card' style='text-align:center; padding:10rem 2rem; color:var(--text-muted);'>AWAITING FORENSIC INPUT...</div>", unsafe_allow_html=True)

    elif menu == "📈 Intelligence Dashboard":
        st.markdown("<div class='dashboard-title'>Advanced Model Analytics</div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Predictive Drivers</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            sns.barplot(x=imp['importances'].head(10).values, y=[f.replace('_', ' ').title() for f in imp['importances'].head(10).index], palette="GnBu_d", ax=ax)
            ax.tick_params(colors='gray')
            for spine in ax.spines.values(): spine.set_edgecolor('gray')
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Validation: ROC Core</div>", unsafe_allow_html=True)
            if os.path.exists("evaluations/roc_curves.png"): st.image("evaluations/roc_curves.png")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Benchmarking Confusion Matrices</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        cms = sorted([f for f in os.listdir("evaluations") if f.startswith("cm_")])
        for i, img in enumerate(cms[:3]):
            cols[i].image(os.path.join("evaluations", img))
        st.markdown("</div>", unsafe_allow_html=True)

    elif menu == "📂 Case Ledger":
        st.markdown("<div class='dashboard-title'>Forensic Case Ledger</div>", unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown("<div class='glass-card' style='text-align:center;'>No records found in current session context.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            df_hist = pd.DataFrame(st.session_state.history)
            df_hist.columns = ["Case ID", "Fraud Probability", "Timestamp", "Event Category", "Aggr. Value ($)"]
            st.dataframe(df_hist.style.format({"Fraud Probability": "{:.1%}", "Aggr. Value ($)": "${:,.0f}"}), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
