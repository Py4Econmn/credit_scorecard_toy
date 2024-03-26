import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
import plotly.express as px

df = pd.read_csv('data\data.csv')


# Binning for age and income, and dummies for education - note that you can use WOE instead of dummies for education as well 
df = pd.get_dummies(df, columns=['educ'], drop_first=True)
df['age_bin'] = pd.cut(df['age'], bins=5)                         # Example: 5 bins for age
df['income_bin'] = pd.qcut(df['income'], q=5, duplicates='drop')  # Example: 5 quantile-based bins for income

# Calculate WoE and IV for age
age_iv_df = pd.DataFrame()
age_iv_df['cnt_default'] = df.groupby('age_bin')['default'].sum()
age_iv_df['cnt_non_default'] = df.groupby('age_bin')['default'].count() - age_iv_df['cnt_default']
age_iv_df['total'] = age_iv_df['cnt_default'] + age_iv_df['cnt_non_default']
age_iv_df['perc_default'] = age_iv_df['cnt_default'] / age_iv_df['total']
age_iv_df['perc_non_default'] = 1 - age_iv_df['perc_default']
age_iv_df['WoE_age'] = np.log(age_iv_df['perc_default'] / age_iv_df['perc_non_default'])
age_iv_df['IV'] = (age_iv_df['perc_default'] - age_iv_df['perc_non_default']) * age_iv_df['WoE_age']
age_iv_df.reset_index(inplace=True)
age_iv = age_iv_df['IV'].sum()


# Calculate WoE and IV for income
income_iv_df = pd.DataFrame()
income_iv_df['cnt_default'] = df.groupby('income_bin')['default'].sum()
income_iv_df['cnt_non_default'] = df.groupby('income_bin')['default'].count() - income_iv_df['cnt_default']
income_iv_df['total'] = income_iv_df['cnt_default'] + income_iv_df['cnt_non_default']
income_iv_df['perc_default'] = income_iv_df['cnt_default'] / income_iv_df['total']
income_iv_df['perc_non_default'] = 1 - income_iv_df['perc_default']
income_iv_df['WoE_income'] = np.log(income_iv_df['perc_default'] / income_iv_df['perc_non_default'])
income_iv_df['IV'] = (income_iv_df['perc_default'] - income_iv_df['perc_non_default']) * income_iv_df['WoE_income']
income_iv_df.reset_index(inplace=True)
income_iv = income_iv_df['IV'].sum()

# Map WoE values back to the original dataset
df = pd.merge(df, income_iv_df[['income_bin', 'WoE_income']], on='income_bin', how='left')
df = pd.merge(df, age_iv_df[['age_bin', 'WoE_age']], on='age_bin', how='left')


features = ['WoE_income','WoE_age','educ_Bachelor','educ_High School','educ_No High School']  # Use WoE values for age and income, and include education
target = 'default'

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)
# Get predicted probabilities for the positive class (class 1)
y_prob = log_reg.predict_proba(X_test)[:, 1]

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", roc_auc)
print(classification_report(y_test, y_pred))

# Calculate false positive rate (fpr) and true positive rate (tpr)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


y_all = df[target]
X_all = df[features]
y_pred_all = log_reg.predict(X_all)
y_prob_all = log_reg.predict_proba(X_all)[:, 1]

roc_auc = roc_auc_score(y_all, y_pred_all)
print("ROC AUC:", roc_auc)
print(classification_report(y_all, y_pred_all))

# Calculate false positive rate (fpr) and true positive rate (tpr)
fpr, tpr, thresholds = roc_curve(y_all, y_prob_all)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# coefficients
intercept = log_reg.intercept_[0]
coefficients = log_reg.coef_[0]
feature_names = X_train.columns.tolist()
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
print("Intercept:", intercept)
print(coefficients_df)

# score 200-800
log_odds = log_reg.intercept_[0] + (df[features] * log_reg.coef_[0]).sum(axis=1)
min_score = 200
max_score = 800
# Linearly scale log-odds to credit score range
credit_score = max_score-((log_odds - log_odds.min()) / (log_odds.max() - log_odds.min())) * (max_score - min_score)
df['Credit_Score'] = credit_score
print(df['Credit_Score'].describe())


# Score distribution by default status
fig = px.histogram(df, x="Credit_Score", color="default",nbins =25,opacity =0.5, barmode="overlay", histnorm="percent")
fig.show()

# PD over score bins
score_intervals = np.linspace(min_score, max_score, num=20)  # Adjust num as needed for desired number of intervals
# Compute histogram counts for default=1 and default=0
default_1 = df[df['Default'] == 1]['Credit_Score']
default_0 = df[df['Default'] == 0]['Credit_Score']
hist_default_1, _ = np.histogram(default_1, bins=score_intervals)
hist_default_0, _ = np.histogram(default_0, bins=score_intervals)
# Calculate percentages
total_counts = hist_default_1 + hist_default_0
percentage_default_1 = (hist_default_1 / total_counts) * 100
# Plot histograms of ratios
plt.figure(figsize=(10, 6))
plt.bar(score_intervals[:-1], percentage_default_1, width=score_intervals[1] - score_intervals[0], color='red', alpha=0.5, label='Default=1')
plt.xlabel('Credit Score')
plt.ylabel('Percentage')
plt.title('Percentage of Defaults by Credit Score Interval')
plt.legend()
plt.grid(True)
plt.show()


# mean_pd = df.groupby('educ')['default'].mean()
# plt.plot(mean_pd.index,mean_pd.values)
# plt.show()
# educ_mapping = {'No High School': 1,'High School': 2,'Bachelor': 3,'Above': 4} 
# df['educ_enc'] = df['educ'].map(educ_mapping)
