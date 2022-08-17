#!/usr/bin/env python
# coding: utf-8

# # final project task 
# 

# Data Analysis on Spotify Million Song Plyalist 

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# In[28]:


track_feat=pd.read_csv("track_feats.csv")


# In[29]:


track_feat


# # 2. Get an overview of the data & Clean it

# In[30]:


# Display an overview of the DataFrame
track_feat.info()


# In[31]:


track_feat.describe().transpose()


# In[38]:


# Allocate target variable
track_feat_y = track_feat.iloc[:,3]
track_feat_y


# In[40]:


# Drop unnecessary columns
track_feat_x = track_feat.drop(columns=["track_id", "us_popularity_estimate"])

# Print unique values of columns
for col in track_feat_x.columns:
    print('{} : {}\n'.format(col,track_feat_x[col].unique()))


# In[32]:


track_feat.isnull().sum()


# In[41]:


# Compute the summary statistics of all columns in the `track_feats` DataFrame
print(track_feat_x.describe())


# Graphical summary of the data 

# In[34]:


plt.figure(figsize=(15,15))
heatmap=sns.heatmap(track_feat.corr(),cbar=True,annot=True, fmt=".1g",vmin=-1,vmax=1,center=0,cmap="magma",linewidths =1)
heatmap.set_title("correlation between variable")
plt.show


# In[35]:


#taking sample from large dataset
sample_df = track_feat.sample(int(0.004*len(track_feat1)))
len(sample_df)


# In[36]:


#relationship of beat_strenght and loudness
plt.figure(figsize=(10,6))
sns.regplot(data=sample_df,y="beat_strength",x="danceability",color="c",marker="*").set(title="beat_strength vs danceability")
plt.show()


# In[20]:


plt.figure(figsize=(10,6))
sns.regplot(data=sample_df,y="tempo",x="loudness",color="c",marker="*").set(title="tempo vs loudness")
plt.show()


# # Fit a multivariate linear regression

# In[42]:


# Create a linear regression object
reg = LinearRegression()

# Fit a multivariate linear regression model
reg.fit(track_feat_x,track_feat_y)

# Retrieve the regression coefficients
for i in range(track_feat_x.shape[1]):
    print(f"Correlation Coefficient of {track_feat_x.columns[i]} is {reg.coef_[i]:.4f}\n")


# So, we notice there is a good correlation between target variable us_popularity_estimate and 
# bounciness
# beat_strength
# acousticness
# danceability
# energy
# acoustic_vector_3
# time_signature
# mechanism

# # Perform PCA on standardized data

# In[44]:


# Standardize and center the feature columns
scaler = StandardScaler()
features_scaled = scaler.fit_transform(track_feat_x)
# Perform PCA to make dimensionality reduction 
pca = PCA()
# Fit the standardized data to the pca
pca=pca.fit(features_scaled)
# Plot the proportion of variance explained on the y-axis of the bar plot
# Set up the matplotlib plot configuration
f, ax = plt.subplots(figsize=(30, 20))
plt.bar(range(1, pca.n_components_ + 1),  pca.explained_variance_ratio_, color='black')
plt.xlabel('Principal component #')
plt.ylabel('Proportion of variance explained')
plt.xticks(range(track_feat_x.shape[1]))
# Compute the cumulative proportion of variance explained by the first two principal components
two_first_comp_var_exp = pca.explained_variance_ratio_.cumsum()[5]
print("The cumulative variance of the first two principal components is {}".format(
    round(two_first_comp_var_exp, 5)))


# # 7. Find clusters of similar states in the data

# In[ ]:


# A loop will be used to plot the explanatory power for up to 12 KMeans clusters
ks = range(1, 12)
inertias = []
for k in ks:
    # Initialize the KMeans object using the current number of clusters (k)
    # Number of time the k-means algorithm will be run with different centroid seeds equals to 50
    km = KMeans(n_clusters=k, random_state=9)
    # Fit the scaled features to the KMeans object
    km.fit(features_scaled)
    # Append the inertia for `km` to the list of inertias
    inertias.append(km.inertia_)
    
# Plot the results in a line plot
plt.plot(ks, inertias, marker='o', color="green")


# 
