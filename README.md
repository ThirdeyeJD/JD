# Employee Attrition Analytics Dashboard

A Streamlit web app that transforms the employee survey dataset into actionable retention insights.

## Tabs & Features
| Tab | Key Capabilities |
|-----|------------------|
| **Data Visualisation** | 10+ descriptive charts, KPIs, and interactive filters |
| **Classification** | K‑NN, Decision Tree, Random Forest & Gradient Boosting; metrics table, labeled confusion matrix, unified ROC curve, batch scoring via CSV upload |
| **Clustering** | K‑means with interactive *k* slider, elbow plot, persona summary table, downloadable cluster‑tagged data |
| **Association Rules** | Apriori mining on multi‑select columns with tunable support/confidence; top‑10 rules table |
| **Regression** | Linear, Ridge, Lasso & Decision‑Tree regressors to surface 5‑7 quick insights |

## Quick Start

```bash
git clone https://github.com/<your‑org>/employee_attrition_dashboard.git
cd employee_attrition_dashboard
pip install -r requirements.txt
streamlit run app.py
```

Deployed on **Streamlit Community Cloud**: simply point the cloud project to this repo & `app.py`.

## Data
`employee_attrition_survey_synthetic.csv` (1 000 rows) ships in the repo. Replace with real HR data (same columns) to go live.