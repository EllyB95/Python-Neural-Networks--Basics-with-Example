# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:25:09 2025

@author: harpr
"""
#%% Improting Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#%%
data = {
    'Feature1': [5, 10, 15, 20, 25, 30, 35, 40],
    'Feature2': [2, 4, 6, 8, 10, 12, 14, 16],
    'Label': [0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

#%%
# Splitting dataset into features and target
X = df[['Feature1', 'Feature2']]
y = df['Label']
#%%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Initialize and train Neural Network model
model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
model.fit(X_train, y_train)
#%%
# Make predictions
y_pred = model.predict(X_test)
print(y_pred)
#%%
# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Neural Network Accuracy: {accuracy:.2f}")
#%%#%%

#%%

#%%

#%%

#%%

