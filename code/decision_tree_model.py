# Note: In development!

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Data
data = pd.read_csv('data\data.csv')
X = data.drop(columns=['default'])
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(transformers=[('onehot', OneHotEncoder(), ['educ'])],remainder='passthrough')),
    ('classifier', DecisionTreeClassifier(random_state=42,max_depth=10))
])
pipeline.fit(X_train,y_train)
y_score = pipeline.predict_proba(X_test)[:, 1]  # PD
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# Score
target_score = 600
target_odds = 2/98 
PDO = 100  

y_score = pipeline.predict_proba(X_train)[:, 1]  # PD

# Calculate log-odds from predicted probabilities
log_odds = np.log((y_score + 1e-6)/ (1 - y_score + 1e-6))  

# Calculate scores using PDO method
score = target_score - (PDO / np.log(2)) * (log_odds - np.log(target_odds))
score = np.clip(score, 200, 800)
plt.hist(score)
plt.show()
