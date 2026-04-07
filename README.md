# Customer Churn Prediction

## Problem Statement

Telecom companies lose 15-25% of customers annually. This project predicts which customers are likely to churn, enabling proactive retention campaigns.

## Technical Approach

- **Data**: IBM Telco Customer Churn dataset (7,043 customers, 21 features)
- **Models**: XGBoost (best), Random Forest, Gradient Boosting
- **Techniques**: SMOTE for imbalance, SHAP for interpretability
- **Deployment**: Streamlit interactive dashboard

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.87 |
| Precision (Churn) | 0.81 |
| Recall (Churn) | 0.76 |
| F1-Score | 0.78 |

## Key Insights

1. **Tenure is critical**: Customers with <6 months tenure are 5x more likely to churn
2. **Contract matters**: Month-to-month contracts have 3x higher churn
3. **Payment method**: Electronic check users churn 2x more
4. **Services gap**: No online security/tech support increases churn by 40%

## Live Demo

[Click here to try the app](http://localhost:8501/)

## Project Structure
├── app.py # Streamlit dashboard
├── notebooks/
│ └── churn_analysis.ipynb
├── models/
│ └── churn_model.pkl
└── requirements.txt

# Local Setup

```bash
git clone https://github.com/yourusername/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py

📈 Business Impact
Potential savings: $5M+ annually for mid-size telecom
ROI: 3-5x retention marketing spend
Actionable: Provides churn probability and top risk factors

🔮 Future Enhancements
Real-time predictions via API
Customer segmentation for targeted campaigns
A/B testing framework for retention offers
