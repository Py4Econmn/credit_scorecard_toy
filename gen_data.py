import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)
n_obs = 10000

income = np.random.uniform(20000, 200000, size=n_obs) # Generate 'Income'
age = np.random.randint(18, 81, size=n_obs)           # Generate 'Age'
# Generate 'Education'
education_levels = ['No High School', 'High School', 'Bachelor', 'Above'] 
education = np.random.choice(education_levels, size=n_obs) 

# Default probability increases with lower income, younger age (up to a certain point), and lower education level
default_probability = 1-(income / 200000) * (1-((age - 50) / (80 - 18))**2) * ((np.array([education_levels.index(x) + 1 for x in education])) / len(education_levels))
# Generate 'Default' column based on calculated default probabilities
defaulted = np.random.binomial(n=1, p=default_probability)

df = pd.DataFrame({
    'Default': defaulted,
    'Income': income,
    'Age': age,
    'Education': education
})

print(df.head())

# age_probabilities = df.groupby('Age')['Default'].mean()
age_probabilities = df.groupby('Age')['Default'].mean()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(age_probabilities.index, age_probabilities.values, marker='o', linestyle='-')
plt.title('Probability of Default by Age')
plt.xlabel('Age')
plt.ylabel('Probability of Default')
plt.grid(True)
plt.show()

