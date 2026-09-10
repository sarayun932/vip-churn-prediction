# 📊 VIP Customer Churn Prediction
### SVM-Based Classification · Business Data Analytics II

> Which VIP customers are about to leave — and why? Built a machine learning classification model to predict churn among high-value bank customers, then translated the findings into a practical early-warning framework for financial institutions.

---

## 🔍 Project Overview

| | |
|---|---|
| **Type** | Individual Academic Project |
| **Course** | Business Data Analytics II: ML-Based Business Model Development |
| **Institution** | Sookmyung Women's University |
| **Period** | Sep 2025 – Dec 2025 |
| **Role** | Solo — full pipeline from EDA to model deployment |
| **Grade** | A+ |

---

## 🛠 Tech & Methods

`R` `SVM` `e1071` `caret` `ggplot2` `corrplot` `Welch's ANOVA` `Confusion Matrix` `Cohen's Kappa` `Hyperparameter Tuning (tune.svm)`

---

## 📌 Problem

VIP customers represent 50% of a bank's customer base but account for over 51% of all churn — a disproportionate revenue risk. Existing retention strategies treat VIP and general customers the same, missing the fact that high-value customers churn for fundamentally different reasons.

**Core question:**
> *"Which VIP customers are likely to churn, and what variables drive that decision?"*

---

## 📂 Data

| | |
|---|---|
| **Source** | Kaggle — Bank Customer Churn Dataset (Kollipara, 2020) |
| **Total size** | 10,000 customers |
| **VIP subset** | 5,002 customers (Diamond + Platinum card holders) |
| **VIP churn rate** | 21.07% (vs. 20.38% overall) |
| **VIP share of total churn** | 51.71% |
| **Variables** | Demographics, financial data, satisfaction score, complaint history, card type, points earned |
| **Target** | Exited (0 = retained, 1 = churned) |

---

## 🔬 Methodology

**01. Exploratory Data Analysis (EDA)**
Profiled VIP customer demographics and financial characteristics. Key observations:
- VIP customers concentrated in ages 30–50 — financially stable, low default risk
- Balance distribution right-skewed — small number of ultra-high-value customers
- Credit scores concentrated in 600–750 range

**02. Churn Pattern Analysis**
Compared churned vs. retained VIP customers across all variables:
- Complaint history (Complain = 1): churn rate **99.5%** vs. **0.02%** for no complaints → near-perfect separator
- Inactive members showed significantly higher churn rates
- Balance and credit score alone could not distinguish churners

**03. Correlation Analysis**
Computed correlation matrix across Age, Balance, CreditScore, Tenure, and Satisfaction Score. No strong linear relationships → confirmed non-linear, multi-variable churn patterns → justified SVM choice.

**04. SVM Modelling — 4 Kernels Compared**
Trained and tuned SVM models using `tune.svm()` with cross-validation:

| Kernel | Accuracy | Kappa | Sensitivity | Specificity |
|---|---|---|---|---|
| Linear | 0.9987 | 0.9959 | 0.9983 | 1.0000 |
| Polynomial | 0.9987 | 0.9959 | 0.9983 | 1.0000 |
| RBF | 0.9987 | 0.9959 | 0.9983 | 1.0000 |
| Sigmoid | 0.9987 | 0.9959 | 0.9983 | 1.0000 |

All kernels performed identically — explained by near-perfect separation created by the Complain variable.

**05. Final Model Selection**
Selected **Linear SVM** — same performance as all others, but simpler structure, lower computational cost, and higher interpretability.

---

## 📊 Key Findings

- VIP churn is driven primarily by **behavioural and experiential signals**, not financial metrics
- A single complaint (Complain = 1) predicts churn with 99.5% accuracy — but this is a **lagging indicator**
- The structural insight: **churn prevention requires detecting pre-complaint signals** — declining activity, falling satisfaction scores
- All 4 kernels achieved identical performance → VIP churn follows a clearly separable, behaviour-driven pattern

---

## 🚀 Business Implications

**Stage 1 — Current model as triage tool**
Use the SVM model to flag at-risk VIP customers based on static indicators (activity status, satisfaction score) before complaints occur.

**Stage 2 — Dynamic prediction system (future work)**
Supplement with time-series behavioural data — transaction frequency trends, digital channel usage logs, balance volatility — to catch churn signals weeks before they become complaints.

---

## ⚠️ Limitations

- The Complain variable may function as a proxy for the outcome itself — future work should test models with Complain excluded
- Cross-sectional data limits causal inference
- Single dataset limits external validity
- SVM interpretability is limited — SHAP values would improve practical usability

---

## 📁 Repository Structure

├── README.md
└── vip_churn_analysis.R     # Full analysis pipeline

---

## 🔗 Links

- 📎 [Portfolio (Notion)](https://www.notion.so/Portfolio-356dc77303348003b532f2ed4fb72183)
  
