#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


# Load the dataset
data = pd.read_csv("data.csv")


# In[4]:


model_size = data["Model_size_GB"].values
inference_time = data["Inference_Time_ms"].values
BLEU_score = data["BLEU_Score"].values
fact_checking_score = data["Fact_Checking_Score_(0-100)"].values


# In[6]:


# Weights for each parameter
weights = np.array([0.3, 0.3, 0.2, 0.2])


# In[7]:


# Normalize the matrix
normalized_matrix = np.column_stack(
    [
        np.max(model_size) / model_size,                 # Minimize (smaller model size is better)
        np.max(inference_time) / inference_time,         # Minimize (lower inference time is better)
        BLEU_score / np.max(BLEU_score),                 # Maximize (higher BLEU score is better)
        fact_checking_score / np.max(fact_checking_score) # Maximize (higher fact-checking score is better)
    ]
)


# In[8]:


# Calculate the weighted normalized decision matrix
weighted_normalized_matrix = normalized_matrix * weights


# In[9]:


# Ideal and Negative Ideal solutions
ideal_solution = np.max(weighted_normalized_matrix, axis=0)
negative_ideal_solution = np.min(weighted_normalized_matrix, axis=0)


# In[10]:


# Calculate the separation measures
distance_to_ideal = np.sqrt(
    np.sum((weighted_normalized_matrix - ideal_solution) ** 2, axis=1)
)
distance_to_negative_ideal = np.sqrt(
    np.sum((weighted_normalized_matrix - negative_ideal_solution) ** 2, axis=1)
)



# In[11]:


# Calculate the TOPSIS scores
topsis_scores = distance_to_negative_ideal / (
    distance_to_ideal + distance_to_negative_ideal
)


# In[12]:


# Rank the models based on TOPSIS scores
data["TOPSIS_Score"] = topsis_scores
data["Rank"] = data["TOPSIS_Score"].rank(ascending=False)


# In[13]:


# Print the results
print("Model Ranking:")
print(data[["Model", "TOPSIS_Score", "Rank"]])


# In[15]:


data.to_csv("result.csv", index=False)


# In[ ]:




