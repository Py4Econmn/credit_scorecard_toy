import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
n_obs = int(1e6)    # 1 million obs

income = np.random.uniform(1, 10, size=n_obs) # 500k - 10 mio MNT
age = np.random.randint(18, 81, size=n_obs)  
educ_mapping = {'No High School': 1,'High School': 2,'Bachelor': 3,'Above': 4}   
educ = np.random.choice(list(educ_mapping.keys()), size=n_obs) # education from above mapping

df = pd.DataFrame({
    'income': income,
    'age': age,
    'educ': educ})

df['educ_enc'] = df['educ'].map(educ_mapping)
income_bins = [0, 2, 4, 6, 10]  # Define income ranges
income_labels = ['0-2m', '2m-4m', '4m-6m', '6m-10m']  # Define labels for each income range
df['income_enc'] = pd.cut(df['income'], bins=income_bins, labels=income_labels, right=False)

# compute and simulate default probabilities
# compute
logit = 2 + 1 * (-((df['age'] - 50) / (80 - 18))**2) + 1 * (df['income'] / 10-0.05) + 0.5* (df['educ_enc']-1)
df['pd_sim'] = 1 / (1 + np.exp(logit))
# simulate
df['default'] = np.random.binomial(n=1, p=df['pd_sim']) 

# see result
mean_pd = df.groupby(['income_enc', 'educ', 'age'])['default'].mean().reset_index()
print(mean_pd)

# Plot - PD by attributes
attributes = ['income_enc', 'age', 'educ']
fig, axes = plt.subplots(1, len(attributes), figsize=(18, 6), sharey=True)
for ax, attribute in zip(axes, attributes):
    mean_probs_attr = df.groupby(attribute)['default'].mean()
    ax.plot(mean_probs_attr.index, mean_probs_attr.values, marker='o')
    ax.set_title(f'Probability of Default vs. {attribute.capitalize()}')
    ax.set_xlabel(attribute.capitalize())
    ax.set_ylabel('Mean Probability of Default')
    ax.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig('figure\data.png')

df[['income','age','educ','default']].to_csv('data\data.csv',index=False)