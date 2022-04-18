#!/usr/bin/env python
# coding: utf-8

# ## Loading Important Libraries 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# ## Loading data

# In[2]:


data = pd.read_csv('Mall_Customers.csv')
data.head()


# In[3]:


data.shape


# In[4]:


data.duplicated().sum()


# In[5]:


data = data.drop_duplicates()


# ## Taking data for Model 

# In[6]:


data = data.loc[:, ['Annual Income (k$)', 'Spending Score (1-100)']]
data.head()


# In[7]:


data.describe()


# ## Getting values of the data 

# In[8]:


X = data.values


# # Visualizing Data 

# In[9]:


sns.scatterplot(X[:,0], X[:, 1])
plt.xlabel('Anual Income')
plt.ylabel('Spending Score')
plt.show()


# # K-Mean Started

# ## Calculating cost

# In[10]:


def calculate_cost(data, centroid, cluster):
  sum = 0
  for i, val in enumerate(data):
    sum += np.sqrt((centroids[int(cluster[i]), 0]-val[0])**2 +(centroids[int(cluster[i]), 1]-val[1])**2)
  return sum


# ## Setting the model with define functions

# In[11]:


def kmeans(data1, k):
    diff = 1
    cluster = np.zeros(data1.shape[0])
    centroids = data.sample(n=k).values
    while diff:
        # Checking for each observation
        for i, row in enumerate(data1):
            mn_dist = float('inf')
        # Euclidean distance from the point
            for idx, centroid in enumerate(centroids):
                d = np.sqrt((centroid[0]-row[0])**2 + (centroid[1]-row[1])**2)
            # Getting the point with lowest distance from centroid
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = idx
        new_centroids = pd.DataFrame(data1).groupby(by=cluster).mean().values
        # Check if repeating centroids
        if np.count_nonzero(centroids-new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids
        return centroids, cluster


# ## Trying with different K values

# In[12]:


cost_list = []
for k in range(1, 10):
    centroids, cluster = kmeans(X, k)
    # sum of square within clusters
    cost = calculate_cost(X, centroids, cluster)
    cost_list.append(cost)


# In[13]:


sns.lineplot(x=range(1,10), y=cost_list, marker='o')
plt.title("Elbow Method")
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Square within Clusters')
plt.show()


# In[14]:


k = 3
centroids, cluster = kmeans(X, k)


# ## Plotting K Clusters 

# In[15]:


sns.scatterplot(X[:,0], X[:, 1], hue=cluster)
sns.scatterplot(centroids[:,0], centroids[:, 1], s=100, color='black')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.show()


# # K-Mean++

# In[16]:


import random as rd
i=rd.randint(0,X.shape[0])
Centroid=np.array([X[i]])


# In[17]:


null =np.array([]) 
for x in X:
    null=np.append(null,np.min(np.sum((x-Centroid)**2)))


# In[18]:


#finding the probability 
Probabilty =null/np.sum(null)


# In[19]:


#comulative probability of the data
cummulative_prob=np.cumsum(Probabilty)


# In[20]:


#doing itterations
r=rd.random()
i=0
for j,p in enumerate(cummulative_prob):
    if r<p:
       i=j
       break
Centroid=np.append(Centroid,[X[i]],axis=0)


# ## Finding the best K value for the model 

# In[21]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Square within Clusters')
plt.show()


# # Visualizing the K-Mean++ Clusters

# In[22]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'magenta', label ='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'grey', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'cyan', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


# # The Elbow Method is showing that the K-Means++  is working better than simple K-mean

# # The End 
