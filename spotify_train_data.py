#!/usr/bin/env python
# coding: utf-8

# # spotify Train_data 

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[7]:


train_data= pd.read_csv("train_data.csv")


# In[8]:


train_data.head()


# In[9]:


train_data.info()


# In[10]:


# Drop unnecessary columns
train_data_x = train_data.drop(columns=["session_id", "track_id_clean"])

# Print unique values of columns
for col in train_data_x.columns:
    print('{} : {}\n'.format(col,train_data_x[col].unique()))


# In[19]:


train_data_x.describe().transpose()


# train_data.isnull()

# In[20]:


train_data.isnull()


# # Quantify the association of features

# In[26]:


plt.figure(figsize=(15,15))
heatmap=sns.heatmap(train_data.corr(),cbar=True,annot=True, fmt=".1g",vmin=-1,vmax=1,center=0,cmap="magma",linewidths =1)
heatmap.set_title("correlation feature")
plt.show


# In[29]:


#taking sample from large dataset
sample_df = train_data.sample(int(0.004*len(train_data)))
len(sample_df)


# In[35]:


#relationship of beat_strenght and loudness
plt.figure(figsize=(10,6))
sns.regplot(data=sample_df,y="no_pause_before_play",x="long_pause_before_play",color="c",marker="*").set(title="no_pause_before_play vs long_pause_before_play")
plt.show()


# #  Perform PCA on standardized data

# In[40]:


# Standardize and center the feature columns
scaler = StandardScaler()
features_scaled = scaler.fit_transform(train_data_x)

# Perform PCA to make dimensionality reduction 
pca = PCA()

# Fit the standardized data to the pca
pca=pca.fit(features_scaled)
# Set up the matplotlib plot configuration
f, ax = plt.subplots(figsize=(20, 15))
plt.bar(range(1, pca.n_components_ + 1),  pca.explained_variance_ratio_, color='green')
plt.xlabel('Principal component #')
plt.ylabel('Proportion of variance explained')
plt.xticks(range(train_data_x.shape[1]))

# Compute the cumulative proportion of variance explained by the first two principal components
two_first_comp_var_exp = pca.explained_variance_ratio_.cumsum()[4]
print("The cumulative variance of the first two principal components is {}".format(
    round(two_first_comp_var_exp, 6)))


# In[ ]:


# A loop will be used to plot the explanatory power for up to 10 KMeans clusters
ks = range(1, 10)
inertias = []
for k in ks:
    # Initialize the KMeans object using the current number of clusters (k)
    # Number of time the k-means algorithm will be run with different centroid seeds equals to 50
    km = KMeans(n_clusters=k, random_state=8)
    # Fit the scaled features to the KMeans object
    km.fit(features_scaled)
    # Append the inertia for `km` to the list of inertias
    inertias.append(km.inertia_)
    
# Plot the results in a line plot
plt.plot(ks, inertias, marker='o', color="green")


# In[ ]:




