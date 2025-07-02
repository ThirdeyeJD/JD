import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_curve, auc,
                             ConfusionMatrixDisplay)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori, association_rules
import io

st.set_page_config(page_title="Attrition Analytics Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("employee_attrition_survey_synthetic.csv")

df = load_data()

st.title("Employee Attrition & Retention – Analytics Suite")

tab_viz, tab_class, tab_cluster, tab_assoc, tab_reg = st.tabs(
    ["📊 Data Visualisation", "🤖 Classification", "🧩 Clustering",
     "🔗 Association Rules", "📈 Regression"])

# --------------------------------------------------------------------
# Data Visualisation
# --------------------------------------------------------------------
with tab_viz:
    st.header("Exploratory Dashboards & KPIs")
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Likelihood to Stay", f"{df['Q1_LikelihoodToStay'].mean():.2f}")
    stay_rate = (df['Q1_LikelihoodToStay'] >= 4).mean()*100
    col2.metric("High-Retention Employees", f"{stay_rate:.1f}%")
    col3.metric("Avg Tenure (yrs)", f"{df['Q7_TenureYears'].mean():.1f}")
    col4.metric("Avg Weekly Hours", f"{df['Q9_WeeklyHours'].mean():.1f}")

    st.markdown("### Key Distributions")
    figs = []
    # 1 Age
    fig1, ax1 = plt.subplots()
    ax1.hist(df["Q2_Age"], bins=20)
    ax1.set_title("Age Distribution")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Count")
    figs.append(fig1)
    # 2 Likelihood
    fig2, ax2 = plt.subplots()
    df["Q1_LikelihoodToStay"].value_counts().sort_index().plot(kind="bar", ax=ax2)
    ax2.set_title("Likelihood to Stay (1–5)")
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Employees")
    figs.append(fig2)
    # 3 Salary band
    fig3, ax3 = plt.subplots()
    df["Q8_SalaryBand"].value_counts().plot(kind="bar", ax=ax3)
    ax3.set_title("Salary Band Distribution")
    figs.append(fig3)
    # 4 Tenure
    fig4, ax4 = plt.subplots()
    ax4.hist(df["Q7_TenureYears"], bins=20)
    ax4.set_title("Tenure Distribution")
    ax4.set_xlabel("Years")
    figs.append(fig4)
    # 5 Weekly hours
    fig5, ax5 = plt.subplots()
    ax5.hist(df["Q9_WeeklyHours"], bins=20)
    ax5.set_title("Weekly Hours Distribution")
    ax5.set_xlabel("Hours/week")
    figs.append(fig5)
    # 6 Department vs stay
    fig6, ax6 = plt.subplots()
    df.groupby("Q6_Department")["Q1_LikelihoodToStay"].mean().plot(kind="bar", ax=ax6)
    ax6.set_title("Avg Stay Likelihood by Dept")
    figs.append(fig6)
    # 7 Work arrangement
    fig7, ax7 = plt.subplots()
    df["Q10_WorkArrangement"].value_counts().plot(kind="pie", autopct="%1.0f%%", ax=ax7)
    ax7.set_ylabel("")
    ax7.set_title("Work Arrangement Split")
    figs.append(fig7)
    # 8 Commute vs likelihood scatter
    fig8, ax8 = plt.subplots()
    ax8.scatter(df["Q11_CommuteMin"], df["Q1_LikelihoodToStay"])
    ax8.set_xlabel("Commute (min)")
    ax8.set_ylabel("Likelihood to Stay")
    ax8.set_title("Commute vs Stay Likelihood")
    figs.append(fig8)
    # 9 Training hours
    fig9, ax9 = plt.subplots()
    ax9.hist(df["Q19_TrainingHours"], bins=20)
    ax9.set_title("Annual Training Hours")
    figs.append(fig9)
    #10 Satisfaction vs Likelihood box
    fig10, ax10 = plt.subplots()
    df.boxplot(column="Q12_JobSatisfaction", by="Q1_LikelihoodToStay", ax=ax10)
    ax10.set_title("Job Satisfaction by Stay Likelihood")
    ax10.set_xlabel("Likelihood to Stay")
    ax10.set_ylabel("Job Satisfaction")
    figs.append(fig10)
    # Show figures
    for f in figs:
        st.pyplot(f)

# --------------------------------------------------------------------
# Helper: preprocessing for models
# --------------------------------------------------------------------
def preprocessor(X):
    cat_cols = X.select_dtypes("object").columns
    num_cols = X.select_dtypes(exclude="object").columns
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# --------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------
with tab_class:
    st.header("Predicting Retention Propensity")
    # Binary target
    X = df.drop(columns=["Q1_LikelihoodToStay"])
    y = (df["Q1_LikelihoodToStay"] >= 4).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosted": GradientBoostingClassifier(random_state=42)
    }
    metrics = []
    pipelines = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
        y_pred = pipe.predict(X_test)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0)
        metrics.append([name,
                        accuracy_score(y_train, pipe.predict(X_train)),
                        accuracy_score(y_test, y_pred),
                        prec, rec, f1])
    met_df = pd.DataFrame(metrics, columns=["Model", "Train Acc", "Test Acc",
                                            "Precision", "Recall", "F1"])
    st.dataframe(met_df.style.format(precision=3), use_container_width=True)

    st.subheader("Confusion Matrix")
    sel_model = st.selectbox("Select algorithm", list(models.keys()))
    cm = confusion_matrix(y_test, pipelines[sel_model].predict(X_test))
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["Leave/Low", "Stay/High"])
    disp.plot(ax=ax_cm, colorbar=False)
    ax_cm.set_title(f"{sel_model} Confusion Matrix")
    st.pyplot(fig_cm)

    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots()
    for name, pipe in pipelines.items():
        y_score = pipe.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr, tpr):.2f})")
    ax_roc.plot([0,1],[0,1],'--')
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.legend()
    st.pyplot(fig_roc)

    st.subheader("Batch Scoring")
    upl = st.file_uploader("Upload CSV (same schema, without Q1_LikelihoodToStay)", type=["csv"])
    if upl:
        new_df = pd.read_csv(upl)
        pred = pipelines[sel_model].predict(new_df)
        res = new_df.copy()
        res["StayPrediction"] = pred
        st.dataframe(res.head())
        buf = io.BytesIO()
        res.to_csv(buf, index=False)
        st.download_button("Download Predictions", data=buf.getvalue(),
                           file_name="attrition_predictions.csv", mime="text/csv")

# --------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------
with tab_cluster:
    st.header("Employee Segmentation – K‑Means")
    k_val = st.slider("Number of clusters", 2, 10, 3)
    df_enc = pd.get_dummies(df.drop(columns=["Q1_LikelihoodToStay"]))
    km = KMeans(n_clusters=k_val, random_state=42, n_init="auto").fit(df_enc)
    df_cluster = df.copy()
    df_cluster["Cluster"] = km.labels_
    # Elbow
    st.subheader("Elbow Plot")
    inertias=[]
    for k in range(2,11):
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(df_enc)
        inertias.append(km_tmp.inertia_)
    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(2,11), inertias, marker='o')
    ax_elbow.set_xlabel("k")
    ax_elbow.set_ylabel("Inertia")
    st.pyplot(fig_elbow)
    # Persona table
    st.subheader("Cluster Persona Snapshot")
    persona = df_cluster.groupby("Cluster").agg(
        MedianAge=("Q2_Age","median"),
        AvgTenure=("Q7_TenureYears","mean"),
        AvgSalary=("Q8_SalaryBand","first"),
        AvgStayScore=("Q1_LikelihoodToStay","mean")
    )
    st.dataframe(persona)
    # Download
    buf_c = io.BytesIO()
    df_cluster.to_csv(buf_c, index=False)
    st.download_button("Download Clustered Data", data=buf_c.getvalue(),
                       file_name="cluster_tagged_data.csv", mime="text/csv")

# --------------------------------------------------------------------
# Association Rules
# --------------------------------------------------------------------
with tab_assoc:
    st.header("Association Rule Mining")
    cols_default = ["Q17_ReasonsLeave", "Q18_MotivatorsStay"]
    cols_select = st.multiselect("Columns to mine", 
                                 ["Q17_ReasonsLeave", "Q18_MotivatorsStay", "Q22_RetentionInitiatives"],
                                 default=cols_default)
    min_sup = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min Confidence", 0.1, 0.9, 0.3, 0.05)
    basket = pd.DataFrame()
    for col in cols_select:
        basket = pd.concat([basket, df[col].str.get_dummies(sep=", ")], axis=1)
    freq = apriori(basket.astype(bool), min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    top10 = rules.sort_values("confidence", ascending=False).head(10)
    st.dataframe(top10[["antecedents","consequents","support","confidence","lift"]])

# --------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------
with tab_reg:
    st.header("Regression Insights")
    target = st.selectbox("Select numerical target",
                          ["Q7_TenureYears", "Q9_WeeklyHours", "Q11_CommuteMin", "Q19_TrainingHours"])
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
    metric_rows=[]
    for name, model in reg_models.items():
        pipe = Pipeline([("prep", preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        r2_train = pipe.score(X_train, y_train)
        r2_test  = pipe.score(X_test, y_test)
        metric_rows.append([name, r2_train, r2_test])
    rdf = pd.DataFrame(metric_rows, columns=["Model","Train R²","Test R²"])
    st.dataframe(rdf.style.format(precision=3), use_container_width=True)
    st.markdown("Higher R² on test indicates stronger predictive power without overfitting.")