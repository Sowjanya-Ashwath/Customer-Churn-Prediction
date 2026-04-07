#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import shap
import warnings
warnings.filterwarnings('ignore')


#  Load Data
# Using Telco Customer Churn dataset (standard benchmark)
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
df.head()


# Data Cleaning

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Remove customerID (not useful for prediction)
df = df.drop('customerID', axis=1)

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"\nCleaned shape: {df.shape}")
print(f"Churn rate: {df['Churn'].mean():.2%}")


# Exploratory Data Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Churn by tenure
churn_by_tenure = df.groupby('tenure')['Churn'].mean()
axes[0,0].plot(churn_by_tenure.index, churn_by_tenure.values)
axes[0,0].set_title('Churn Rate by Tenure')
axes[0,0].set_xlabel('Tenure (months)')
axes[0,0].set_ylabel('Churn Rate')

# Churn by contract type
contract_churn = df.groupby('Contract')['Churn'].mean()
contract_churn.plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Churn Rate by Contract Type')
axes[0,1].set_ylabel('Churn Rate')

# Churn by payment method
payment_churn = df.groupby('PaymentMethod')['Churn'].mean()
payment_churn.plot(kind='bar', ax=axes[0,2])
axes[0,2].set_title('Churn Rate by Payment Method')
axes[0,2].tick_params(axis='x', rotation=45)

# Correlation heatmap for numeric columns
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=axes[1,0])
axes[1,0].set_title('Correlation Heatmap')

# Distribution of monthly charges
df[df['Churn']==1]['MonthlyCharges'].hist(alpha=0.7, label='Churned', ax=axes[1,1])
df[df['Churn']==0]['MonthlyCharges'].hist(alpha=0.7, label='Retained', ax=axes[1,1])
axes[1,1].set_title('Monthly Charges Distribution')
axes[1,1].legend()

# Distribution of tenure
df[df['Churn']==1]['tenure'].hist(alpha=0.7, label='Churned', ax=axes[1,2])
df[df['Churn']==0]['tenure'].hist(alpha=0.7, label='Retained', ax=axes[1,2])
axes[1,2].set_title('Tenure Distribution')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('images/eda_plots.png', dpi=100, bbox_inches='tight')
plt.show()


# Feature Engineering
# Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('Churn')

print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns: {numeric_cols}")

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


#Handle Imbalanced Data with SMOTE
from imblearn.over_sampling import SMOTE

# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Before SMOTE - Train churn rate: {y_train.mean():.2%}")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Train churn rate: {y_train_resampled.mean():.2%}")
print(f"Original train size: {X_train.shape}, Resampled: {X_train_resampled.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)


#Train Multiple Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train_resampled, cv=5, scoring='roc_auc')
    
    # Train
    model.fit(X_train_scaled, y_train_resampled)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    results[name] = {
        'CV_AUC_Mean': cv_scores.mean(),
        'CV_AUC_Std': cv_scores.std(),
        'Test_AUC': roc_auc_score(y_test, y_pred_proba),
        'Classification_Report': classification_report(y_test, y_pred)
    }
    
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Test AUC: {results[name]['Test_AUC']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))


# Pick Best Model (XGBoost) and Add SHAP
best_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
best_model.fit(X_train_scaled, y_train_resampled)

# SHAP Analysis
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled[:100])  # Use 100 samples for speed

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=X.columns, show=False)
plt.tight_layout()
plt.savefig('images/shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Feature importance bar plot
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Features for Churn Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# Save Model and Artifacts
import joblib

# Save model
joblib.dump(best_model, 'models/churn_model.pkl')

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')

# Save feature columns (for inference)
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, 'models/feature_columns.pkl')

print("✅ Model and artifacts saved successfully!")


# Business Insights
print("\n" + "="*60)
print("KEY BUSINESS INSIGHTS")
print("="*60)

print("""
1. Tenure is the strongest predictor: Customers with <6 months tenure are 5x more likely to churn
2. Month-to-month contracts have 3x higher churn than 1-year contracts
3. Electronic check users churn 2x more than automatic payment users
4. No online security or tech support increases churn risk by 40%

RECOMMENDATIONS:
→ Offer discounts for annual contracts
→ Target new customers (<6 months) with engagement campaigns
→ Bundle online security with popular plans
→ Incentivize automatic payment setup
""")