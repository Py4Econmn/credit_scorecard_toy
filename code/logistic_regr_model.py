import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
import plotly.express as px
import seaborn as sns


def woe_iv(df,xvar,yvar='default'):
    # Calculate WOE and IV for xvar, yvar is default
    woename = 'WOE_'+ xvar
    total_default = df['default'].sum()
    total_non_default = len(df) - total_default
    df_woe = pd.DataFrame()
    df_woe['cnt_def']    = df.groupby(xvar)[yvar].sum()
    df_woe['cnt_nondef'] = df.groupby(xvar)[yvar].count() - df_woe['cnt_def']
    df_woe['cnt_total']  = df_woe['cnt_def'] + df_woe['cnt_nondef']
    df_woe['pct_def']    = df_woe['cnt_def'] / total_default
    df_woe['pct_nondef'] = df_woe['cnt_nondef'] / total_non_default
    df_woe[woename]      = np.log(df_woe['pct_nondef']/df_woe['pct_def'])
    df_woe['IV']         = (df_woe['pct_nondef'] - df_woe['pct_def']) * df_woe[woename]
    df_woe.reset_index(drop=False,inplace=True)
    iv                   = df_woe['IV'].sum()

    return [df_woe, iv]


df = pd.read_csv('data\data.csv')

## Binning and computing WOE, IV
df['age_bin'] = pd.cut(df['age'], bins=5)                         # Example: 5 bins for age
df['income_bin'] = pd.qcut(df['income'], q=5, duplicates='drop')  # Example: 5 quantile-based bins for income


attributes = ['income_bin', 'age_bin', 'educ']
fig, axes = plt.subplots(1, len(attributes), figsize=(18, 6), sharey=True)
for ax, attribute in zip(axes, attributes):
    mean_probs_attr = df.groupby(attribute)['default'].mean()
    ax.plot(mean_probs_attr.index.astype(str), mean_probs_attr.values, marker='o')
    ax.set_title(f'Probability of Default vs. {attribute.capitalize()}')
    ax.set_xlabel(attribute.capitalize())
    ax.set_ylabel('Mean Probability of Default')
    ax.grid(True)
plt.tight_layout()
plt.show()

df_income_woe, income_iv = woe_iv(df,'income_bin')
df_age_woe, age_iv       = woe_iv(df,'age_bin')
df_educ_woe, educ_iv     = woe_iv(df,'educ')

# Map WoE values back to the original dataset
df = pd.merge(df, df_income_woe[['income_bin', 'WOE_income_bin']], on='income_bin', how='left')
df = pd.merge(df, df_age_woe[['age_bin', 'WOE_age_bin']], on='age_bin', how='left')
df = pd.merge(df, df_educ_woe[['educ', 'WOE_educ']], on='educ', how='left')


## LOGISTIC REGRESSION MODEL
features = ['WOE_income_bin','WOE_age_bin','WOE_educ']  # Use WoE values for age and income, and include education
target = 'default'

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1] # # Get predicted probabilities for the positive class (class 1)
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", roc_auc)

# Plot ROC curve: false positive rate (fpr) and true positive rate (tpr)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


## SCORECARD: Compute the credit score using PDO, points, odds
coefficients = log_reg.coef_[0]
intercept = log_reg.intercept_[0]

pdo = 100               # points-to-double-the-odds - 700 has twice lower PD than 600, same with 600 vs 500
target_odds  = 2/98
target_score = 600      # Score will be 600 when 0.02/(1-0.02) = PD/(1-PD) is target_odds. Around 2% PD at 600
factor = pdo/np.log(2)
offset = target_score + factor*np.log(target_odds)

df_income_woe['income_points'] = offset/len(features)-factor*(df_income_woe['WOE_income_bin']*coefficients[0] + intercept/len(features))
df_age_woe['age_points']      = offset/len(features)-factor*(df_age_woe['WOE_age_bin']*coefficients[1] + intercept/len(features))
df_educ_woe['educ_points']    = offset/len(features)-factor*(df_educ_woe['WOE_educ']*coefficients[2] + intercept/len(features))
df = pd.merge(df, df_income_woe[['income_bin', 'income_points']], on='income_bin', how='left')
df = pd.merge(df, df_age_woe[['age_bin', 'age_points']], on='age_bin', how='left')
df = pd.merge(df, df_educ_woe[['educ', 'educ_points']], on='educ', how='left')
df['score'] = df[['income_points','age_points','educ_points']].sum(axis=1)


## FIGURES

# score distribution by default status
fig = px.histogram(df, x="score", color="default",nbins =25,opacity =0.5, barmode="overlay", histnorm="percent")
fig.update_layout(title="Distribution of Score by Default Value")
fig.show()

# share of defaults over score bins
df['score_bin'] = pd.cut(df['score'], bins=10)
bin_totals = df.groupby('score_bin').size()
bin_defaults = df.groupby('score_bin')['default'].sum()
bin_ratio = bin_defaults / bin_totals

fig = px.bar(x=bin_ratio.index.astype(str), y=bin_ratio.values, labels={'x': 'Score', 'y': 'Share of Defaults'},
             title='Ratio of Defaults by Scores')
fig.show()

# score distribution by attributes
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.boxplot(x='age_bin', y='score', data=df, ax=axes[0])
sns.boxplot(x='income_bin', y='score', data=df, ax=axes[1])
sns.boxplot(x='educ', y='score', data=df, ax=axes[2])
axes[0].set_title('Age')
axes[1].set_title('Income')
axes[2].set_title('Education')
plt.suptitle('Distribution of Scores')
plt.tight_layout()
plt.show()


## SCORE TABLE
dfs = [df_income_woe, df_age_woe, df_educ_woe]
indicators = ['income', 'age', 'educ']
selected_dfs = []
for df, indicator in zip(dfs, indicators):
    df['indicator'] = indicator
    selected_df = df[[f'{indicator}_bin' if not indicator == 'educ' else indicator, f'{indicator}_points','indicator']]
    selected_df.columns = ['bins', 'points', 'indicator']
    selected_dfs.append(selected_df)
score_table = pd.concat(selected_dfs, ignore_index=True)
print(score_table)

score_table.to_csv('report\score_table.csv')



# References
#https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
# https://medium.com/@yanhuiliu104/credit-scoring-scorecard-development-process-8554c3492b2b
# https://www.mathworks.com/help/finance/case-study-for-a-credit-scorecard-analysis.html?fbclid=IwAR3yNWJUI3-pa7qM3rgx1uwD1GBbaii_HNQJMFUZBG-Zs449CaEE9GXKDWg_aem_AR4zfT1Urx8unHG0ruKUGTs7AQT_e59E5aNbQTtBisY4fXwir8e3MR6eNcC3YY0anrMtVgr3C90yZ00lCYAU0hkc
# https://www.mathworks.com/help/finance/creditscorecard.formatpoints.html?fbclid=IwAR3Z1c49N2voC5YHZtGDna6gjPTQxWBDupGCqrIyrhK78CDQuXCnVgglz20_aem_AR4dYH9hPF53JUlG1GXC2LSCOQ_-UFPylGoAVYPaUSH8CgxRRsizQxzpDQKStEJSVBNaYgnjE6GNqSWFDUu9OXg3#d126e270638
# https://towardsdatascience.com/intro-to-credit-scorecard-9afeaaa3725f
# https://pypi.org/project/scorecardpy/
# https://cran.r-project.org/web/packages/scorecard/scorecard.pdf

# Thanks to OpenAI for sharing ChatGPT! 

