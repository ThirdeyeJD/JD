import numpy as np
import pandas as pd
import random, os, ace_tools
from pathlib import Path

# ----------------------------- CONFIG -------------------------------- #
random.seed(42)
np.random.seed(42)
N_RESP = 1_000   # number of synthetic employees

LIKERT       = [1, 2, 3, 4, 5]
LIKERT_W_POS = [0.05, 0.15, 0.25, 0.35, 0.20]              # mildly positive
LIKERT_W_NEG = LIKERT_W_POS[::-1]                          # inverse skew

# Demographic / categorical distributions
gender_opts = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
gender_w    = [0.6, 0.35, 0.03, 0.02]

edu_opts = ['Diploma', 'Bachelor’s', 'Master’s', 'Doctorate', 'Other']
edu_w    = [0.1, 0.55, 0.25, 0.05, 0.05]

job_levels = ['Entry', 'Mid', 'Senior', 'Lead', 'Management']
job_lv_w   = [0.2, 0.4, 0.25, 0.1, 0.05]

departments = ['Engineering', 'Product', 'Sales', 'Marketing', 'HR',
               'Finance', 'IT Support', 'Other']
dept_w      = [0.35, 0.1, 0.15, 0.1, 0.08, 0.07, 0.1, 0.05]

salary_bands = ['<75k', '75–120k', '120–180k', '180–250k', '>250k']
salary_w_by_level = {
    'Entry'     : [0.6, 0.3, 0.08, 0.02, 0.0],
    'Mid'       : [0.2, 0.4, 0.3, 0.08, 0.02],
    'Senior'    : [0.05, 0.2, 0.4, 0.25, 0.1],
    'Lead'      : [0.02, 0.08, 0.3, 0.4, 0.2],
    'Management': [0.0, 0.05, 0.15, 0.4, 0.4]
}

work_arr_opts = ['On-site', 'Hybrid', 'Fully Remote']
work_arr_w    = [0.5, 0.3, 0.2]

# Multi‑select options
reasons_leave = ['Low pay', 'Limited growth', 'Workload stress', 'Manager conflict',
                 'Commute', 'Lack of flexibility', 'Benefits', 'Culture misfit', 'Other']
motivate_stay = ['Competitive pay', 'Growth path', 'Flexible work', 'Recognition',
                 'Team culture', 'Benefits', 'Purposeful work', 'Manager support', 'Other']
retain_init   = ['Salary review', 'Flexible hours', 'Remote option', 'Mentorship',
                 'Career roadmap', 'Wellness programs', 'Childcare support', 'Other']

def pick_likert(pos=True):
    """Draw a Likert score; pos=True for positive skew, False for negative."""
    return np.random.choice(LIKERT, p=LIKERT_W_POS if pos else LIKERT_W_NEG)

def pick_multi(options, min_k=1, max_k=3):
    k = random.randint(min_k, max_k)
    return ", ".join(random.sample(options, k))

records = []
for _ in range(N_RESP):
    # Demographics -----------------------------------------------------
    age = int(np.clip(np.random.normal(34, 8), 20, 60))
    if random.random() < 0.01:  # outlier
        age = random.randint(61, 70)

    gender = np.random.choice(gender_opts, p=gender_w)
    edu    = np.random.choice(edu_opts, p=edu_w)
    job_lv = np.random.choice(job_levels, p=job_lv_w)
    dept   = np.random.choice(departments, p=dept_w)

    # Tenure (gamma skew) & outliers
    tenure = round(max(0.1, np.random.gamma(shape=1.8, scale=2.5)), 1)
    if random.random() < 0.01:
        tenure = round(np.random.uniform(15, 25), 1)

    # Salary – tied to job level
    salary_band = np.random.choice(salary_bands, p=salary_w_by_level[job_lv])

    # Work hours & arrangement
    weekly_hours = int(np.clip(np.random.normal(45, 8), 30, 70))
    if random.random() < 0.02:
        weekly_hours = random.randint(71, 90)  # overwork outlier

    work_arr = np.random.choice(work_arr_opts, p=work_arr_w)

    commute = 0
    if work_arr != 'Fully Remote':
        commute = int(np.clip(np.random.normal(30, 12), 5, 90))
        if random.random() < 0.02:
            commute = random.randint(91, 150)  # long commute outlier

    # Satisfaction dimensions
    job_sat      = pick_likert(pos=True)
    wl_balance   = pick_likert(pos=False if weekly_hours > 55 else True)
    career_sat   = pick_likert(pos=True)
    manager_rel  = pick_likert(pos=True)
    culture_align= pick_likert(pos=True)

    # Multi‑select questions
    leave_reasons  = pick_multi(reasons_leave, 1, 3)
    stay_motivators= pick_multi(motivate_stay, 1, 3)

    # Training hours
    train_hours = int(np.clip(np.random.normal(20, 10), 0, 60))
    if random.random() < 0.01:
        train_hours = random.randint(100, 300)  # heavy learner outlier

    # eNPS & switch
    recommend = 'Yes' if random.random() < 0.7 else 'No'
    switch15  = np.random.choice(['Yes', 'Maybe', 'No'], p=[0.35, 0.4, 0.25])

    retention_choices = pick_multi(retain_init, 1, 3)

    # Likelihood to stay – correlated with satisfaction & balance
    base = (job_sat + wl_balance + career_sat + manager_rel + culture_align) / 5
    noise = np.random.normal(0, 0.5)
    likelihood = int(np.clip(round(base + noise), 1, 5))

    # Record assembly
    records.append({
        "Q1_LikelihoodToStay": likelihood,
        "Q2_Age": age,
        "Q3_Gender": gender,
        "Q4_Education": edu,
        "Q5_JobLevel": job_lv,
        "Q6_Department": dept,
        "Q7_TenureYears": tenure,
        "Q8_SalaryBand": salary_band,
        "Q9_WeeklyHours": weekly_hours,
        "Q10_WorkArrangement": work_arr,
        "Q11_CommuteMin": commute,
        "Q12_JobSatisfaction": job_sat,
        "Q13_WorkLifeBalance": wl_balance,
        "Q14_CareerGrowthSat": career_sat,
        "Q15_ManagerRelationship": manager_rel,
        "Q16_CultureAlignment": culture_align,
        "Q17_ReasonsLeave": leave_reasons,
        "Q18_MotivatorsStay": stay_motivators,
        "Q19_TrainingHours": train_hours,
        "Q20_Recommend": recommend,
        "Q21_SwitchFor15Raise": switch15,
        "Q22_RetentionInitiatives": retention_choices
    })

# ----------------------- DataFrame & Dtypes ------------------------------ #
df = pd.DataFrame(records)

int_cols = ["Q1_LikelihoodToStay", "Q2_Age", "Q9_WeeklyHours", "Q11_CommuteMin",
            "Q12_JobSatisfaction", "Q13_WorkLifeBalance", "Q14_CareerGrowthSat",
            "Q15_ManagerRelationship", "Q16_CultureAlignment"]
df[int_cols] = df[int_cols].astype("int16")
df["Q7_TenureYears"]          = df["Q7_TenureYears"].astype("float32")
df["Q19_TrainingHours"]       = df["Q19_TrainingHours"].astype("int16")

# ------------------------ Persist to CSV --------------------------------- #
out_path = Path("/mnt/data/employee_attrition_survey_synthetic.csv")
df.to_csv(out_path, index=False, encoding="utf-8-sig")

# ------------------------ Display Preview -------------------------------- #
ace_tools.display_dataframe_to_user(name="Synthetic Employee Attrition Survey – preview",
                                    dataframe=df.head(12))

print(f"✅ Generated {df.shape[0]} rows → {out_path}")
