import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from groq import Groq
import os

class ExplainabilityTrinity:
    def __init__(self):
        # Load models and explainers
        self.models = {}
        self.scalers = {}
        self.explainers = {}
        self.feature_names = joblib.load("models/feature_names.pkl")
        
        # Initialize LLM client
        self.llm_client = None
        if os.getenv("GROQ_API_KEY"):
            self.llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        diseases = ['diabetes', 'heart', 'kidney']
        for disease in diseases:
            try:
                self.models[disease] = joblib.load(f"models/{disease}_model.pkl")
                self.scalers[disease] = joblib.load(f"models/{disease}_scaler.pkl")
                self.explainers[disease] = joblib.load(f"models/{disease}_explainer.pkl")
            except:
                print(f"Warning: Could not load {disease} artifacts")
    
    def get_shap_explanation(self, disease_name, patient_data):
        """Level 1: SHAP-based technical explanation"""
        if disease_name not in self.explainers:
            return None
        
        explainer = self.explainers[disease_name]
        model = self.models[disease_name]
        scaler = self.scalers[disease_name]
        
        # Prepare features
        features_df = self._prepare_features(patient_data, disease_name)
        if features_df is None:
            return None
        
        # Scale features
        scaled_features = scaler.transform(features_df)
        
        # Get SHAP values
        shap_values = explainer.shap_values(scaled_features)
        
        # If it's a list (for multi-class), take the first class
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Create feature importance dataframe
        feature_importance = []
        for i, feature in enumerate(features_df.columns):
            importance = {
                'feature': feature,
                'contribution': shap_values[0][i],
                'value': features_df[feature].iloc[0]
            }
            feature_importance.append(importance)
        
        # Sort by absolute contribution
        feature_importance.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return {
            'feature_importance': feature_importance,
            'shap_values': shap_values[0],
            'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0.5
        }
    
    def get_natural_language_explanation(self, disease_name, risk_score, feature_importance):
        """Level 2: LLM-generated patient-friendly explanation"""
        if not self.llm_client:
            return "LLM explanation not available. Please set GROQ_API_KEY environment variable."
        
        # Get top contributing factors
        top_factors = feature_importance[:3]
        
        # Create prompt
        prompt = f"""You are a medical communicator explaining health risk factors to a patient. 
Explain these risk factors in simple, actionable terms for a {disease_name} risk assessment.

Patient Risk Score: {risk_score:.1%}

Top Contributing Factors:
{chr(10).join([f"- {factor['feature']}: {factor['value']:.1f} (contribution: {factor['contribution']:+.3f})" for factor in top_factors])}

Write 2-3 sentences that a patient can easily understand. Focus on:
1. What the main risk drivers are
2. Why they matter for health
3. Use simple, non-technical language
4. Be encouraging but honest

Keep it under 100 words total."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def get_counterfactual_explanation(self, disease_name, patient_data):
        """Level 3: Counterfactual analysis - 'what would it take to be low risk?'"""
        if disease_name not in self.models:
            return None
        
        model = self.models[disease_name]
        scaler = self.scalers[disease_name]
        
        # Prepare features
        features_df = self._prepare_features(patient_data, disease_name)
        if features_df is None:
            return None
        
        # Get current prediction
        scaled_features = scaler.transform(features_df)
        current_risk = model.predict_proba(scaled_features)[0][1]
        
        # If already low risk, return early
        if current_risk < 0.4:
            return {
                'message': 'Already low risk',
                'changes_needed': [],
                'target_risk': current_risk
            }
        
        # Find minimal changes to achieve low risk (<40%)
        counterfactuals = []
        target_risk = 0.35
        
        # Try modifying each feature individually
        for feature in features_df.columns:
            if features_df[feature].iloc[0] == 0:
                continue  # Skip zero features
            
            original_value = features_df[feature].iloc[0]
            
            # Try different values
            if 'BMI' in feature or 'bmi' in feature.lower():
                test_values = np.linspace(original_value - 5, original_value + 2, 20)
            elif 'Glucose' in feature or 'glucose' in feature.lower() or 'bgr' in feature.lower():
                test_values = np.linspace(max(70, original_value - 30), original_value + 10, 20)
            elif 'Age' in feature or 'age' in feature.lower():
                test_values = np.linspace(original_value - 10, original_value, 20)
            else:
                test_values = np.linspace(original_value * 0.7, original_value * 1.3, 20)
            
            for test_value in test_values:
                if test_value <= 0:
                    continue
                
                # Create counterfactual
                cf_features = features_df.copy()
                cf_features[feature] = test_value
                
                # Predict risk
                scaled_cf = scaler.transform(cf_features)
                cf_risk = model.predict_proba(scaled_cf)[0][1]
                
                if cf_risk < target_risk:
                    counterfactuals.append({
                        'feature': feature,
                        'original_value': original_value,
                        'target_value': test_value,
                        'change': test_value - original_value,
                        'percent_change': (test_value - original_value) / original_value * 100,
                        'new_risk': cf_risk,
                        'risk_reduction': current_risk - cf_risk
                    })
                    break  # Found minimal change for this feature
        
        # Sort by smallest change
        counterfactuals.sort(key=lambda x: abs(x['percent_change']))
        
        return {
            'current_risk': current_risk,
            'target_risk': target_risk,
            'changes_needed': counterfactuals[:3],  # Top 3 easiest changes
            'summary': self._generate_counterfactual_summary(counterfactuals[:1])
        }
    
    def _generate_counterfactual_summary(self, top_counterfactual):
        """Generate a human-readable summary of counterfactual analysis"""
        if not top_counterfactual:
            return "Significant lifestyle changes would be needed to reduce risk."
        
        cf = top_counterfactual[0]
        
        feature_name = cf['feature'].replace('_', ' ').title()
        
        if cf['percent_change'] > 0:
            direction = "increase"
        else:
            direction = "decrease"
        
        summary = (f"To achieve low risk (<{cf['target_risk']:.0%}), "
                   f"{direction} {feature_name} from {cf['original_value']:.1f} "
                   f"to {cf['target_value']:.1f} "
                   f"(change: {cf['change']:+.1f}, "
                   f"risk reduction: {cf['risk_reduction']:.1%})")
        
        return summary
    
    def _prepare_features(self, features_dict, disease_name):
        """Prepare features for the specific disease model"""
        try:
            # Get expected feature names for this disease
            expected_features = self.feature_names[disease_name]
            
            # Create DataFrame with all required features
            patient_data = {}
            for feature in expected_features:
                if feature in features_dict:
                    patient_data[feature] = features_dict[feature]
                else:
                    # Set default value for missing features
                    patient_data[feature] = 0
            
            return pd.DataFrame([patient_data])
        except:
            return None
    
    def create_shap_waterfall_chart(self, disease_name, patient_data):
        """Create SHAP waterfall visualization"""
        shap_data = self.get_shap_explanation(disease_name, patient_data)
        if not shap_data:
            return None
        
        feature_importance = shap_data['feature_importance']
        base_value = shap_data['base_value']
        
        # Prepare data for waterfall chart
        features = []
        values = []
        colors = []
        
        # Start with base value
        current_value = base_value
        
        # Add features in order of contribution
        for factor in feature_importance[:8]:  # Top 8 features
            features.append(factor['feature'])
            values.append(factor['contribution'])
            colors.append('red' if factor['contribution'] > 0 else 'blue')
            current_value += factor['contribution']
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Calculate cumulative values
        cumulative = [base_value]
        for value in values:
            cumulative.append(cumulative[-1] + value)
        
        # Create bars
        x_positions = list(range(len(features)))
        
        for i, (feature, value, color) in enumerate(zip(features, values, colors)):
            # Bar
            fig.add_trace(go.Bar(
                x=[i],
                y=[value],
                name=feature,
                marker_color=color,
                width=0.6,
                hovertemplate=f'{feature}<br>Contribution: {value:+.3f}<extra></extra>'
            ))
        
        # Add base value line
        fig.add_hline(y=base_value, line_dash="dash", line_color="gray", 
                     annotation_text=f"Base: {base_value:.3f}")
        
        # Update layout
        fig.update_layout(
            title=f'SHAP Waterfall Plot - {disease_name.title()} Risk',
            xaxis_title='Features',
            yaxis_title='Contribution to Risk',
            showlegend=False,
            height=600,
            xaxis={'categoryorder': 'array', 'categoryarray': features}
        )
        
        return fig
    
    def get_comprehensive_explanation(self, disease_name, patient_data):
        """Get all three levels of explanation"""
        
        # Get current risk prediction
        model = self.models[disease_name]
        scaler = self.scalers[disease_name]
        features_df = self._prepare_features(patient_data, disease_name)
        
        if features_df is None:
            return None
        
        scaled_features = scaler.transform(features_df)
        risk_score = model.predict_proba(scaled_features)[0][1]
        
        # Level 1: SHAP explanation
        shap_explanation = self.get_shap_explanation(disease_name, patient_data)
        
        # Level 2: Natural language explanation
        nl_explanation = self.get_natural_language_explanation(
            disease_name, risk_score, shap_explanation['feature_importance'] if shap_explanation else []
        )
        
        # Level 3: Counterfactual explanation
        counterfactual_explanation = self.get_counterfactual_explanation(disease_name, patient_data)
        
        return {
            'risk_score': risk_score,
            'risk_category': self._get_risk_category(risk_score),
            'shap_explanation': shap_explanation,
            'natural_language_explanation': nl_explanation,
            'counterfactual_explanation': counterfactual_explanation
        }
    
    def _get_risk_category(self, risk_score):
        """Categorize risk score"""
        if risk_score < 0.4:
            return "Low Risk"
        elif risk_score < 0.7:
            return "Moderate Risk"
        else:
            return "High Risk"

if __name__ == "__main__":
    # Test the explainability system
    explainer = ExplainabilityTrinity()
    
    # Example patient data
    patient_data = {
        'Pregnancies': 2,
        'Glucose': 145,
        'BloodPressure': 80,
        'SkinThickness': 20,
        'Insulin': 85,
        'BMI': 32.5,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 45
    }
    
    # Get comprehensive explanation
    explanation = explainer.get_comprehensive_explanation('diabetes', patient_data)
    
    print(f"Risk Score: {explanation['risk_score']:.3f}")
    print(f"Risk Category: {explanation['risk_category']}")
    print(f"Natural Language: {explanation['natural_language_explanation']}")
    
    if explanation['counterfactual_explanation']:
        print(f"Counterfactual: {explanation['counterfactual_explanation']['summary']}")
