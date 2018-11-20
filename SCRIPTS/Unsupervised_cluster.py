# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 12:08:48 2018

@author: kennedy
"""
from os import chdir
import pandas as pd
import matplotlib.pyplot as plt
from cluster_sheets import Master_sheet

#set directory
chdir('D:\\GIT PROJECT\\CLUSTERING')

#load Excel workbook
dataset = pd.ExcelFile('researchlinedata.xlsx')


df_CC, df_BR, df_Mea, df_Peos, df_Wilms = Master_sheet.dataframe(dataset)
#preprocess the sheets
df_BR = Master_sheet.preprocess(df_BR)
df_CC = Master_sheet.preprocess(df_CC)
df_Mea = Master_sheet.preprocess(df_Mea)
df_Peos = Master_sheet.preprocess(df_Peos)
df_Wilms = Master_sheet.preprocess(df_Wilms)


#%% SAVE DATASET


'''Save the datasets'''

df_BR.to_csv('df_BR.csv')
df_CC.to_csv('df_CC.csv')
df_Mea.to_csv('df_Mea.csv')
df_Peos.to_csv('df_Peos.csv')
df_Wilms.to_csv('df_Wilms.csv')

#%% ploT OF OPTIMUM CLUSTER


#plt.plot(df_BR.iloc[:, 0].values, df_BR.iloc[:, 2].values)
#cluster algorithm...
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
#KMeans class


def plot_optimum_cluster(data):
  #set a list to append the iter values
  iter_num = []
  for i in range(1, 15):
      #perform kmeans to get best cluster value using elbow method
      kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42, max_iter = 300)
      kmeans.fit(data)
      iter_num.append(kmeans.inertia_)
      #plot the optimum graph
  plt.plot(range(1, 15), iter_num)
  plt.title('The Elbow method of determining number of clusters')
  plt.xlabel('Number of clusters')
  plt.ylabel('iter_num')
  plt.show()

'''
The optimum cluster for our dataset according to 
cluster the dataset is 4
'''

plot_optimum_cluster(df_Wilms.iloc[:, 1:])



#%% plot of scalled dataset
from sklearn.preprocessing import StandardScaler

'''Note here that the mean is centered around zero
We plot all the dataset using scatter plot function 
from matplotlib'''



features = list(df_Wilms.iloc[:, 1:].columns)

#seperating the features from the target
dX = df_Wilms.iloc[:, 1:]
dY = df_BR['Device']

#statndardize the X's
dX = StandardScaler().fit_transform(dX)
dX_frame = pd.DataFrame(dX, columns = features)

for ii in range(len(features)):
  plt.scatter(df_BR['Device'], dX_frame.iloc[:, ii], s = 7)
#  plt.bar(df_BR['Device'], dX_frame.iloc[:, ii])
plt.xlabel('Device name')
plt.ylabel('Standard Scalar of the Xs => (X - Xm)/sd(X)')
plt.title('Scatter plot of Features against Device')



#%% HIERARCHICAL CLUSTERING

from scipy.cluster.hierarchy import dendrogram, linkage  

#check the linkages
linkage = linkage(dX_frame, 'ward')
#device names
labels = features
#labels = list(df_BR['Device'])
#create a fixed plot size
plt.figure(figsize = (12, 12))
#draw the dendograms
dendrogram(linkage,
           orientation='top',
           labels=labels,
           distance_sort= 'descending',
           show_leaf_counts= True)

plt.title('Dendogram df_Wilms representing possible device clustering for unstandardized data:: method == ward')


'''
Anglomerative 
clustering'''

#from sklearn.cluster import AgglomerativeClustering
#import numpy as np
#cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
#
#ymeans = cluster.fit(dX_frame)

#for ii in range(0, 4):
#  plt.scatter(np.where(ymeans == ii), dX_frame.iloc[ymeans == ii, 0], s = 25)
  
  
#%% CORRELATION COEFFICIENTS

import numpy as np


features = list(df_BR.iloc[:, 1:].columns)
#statndardize the X's
dX = StandardScaler().fit_transform(dX)
dX_frame = pd.DataFrame(dX, columns = features)

#plot the standardized dataset
dX_frame.plot()

#get data correlation
corr = dX_frame.corr()

#plot heapmap of dataset
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
#create ticks of dataset columms
ticks = np.arange(0,len(dX_frame.columns),1)
#set axis to tick name i.ie vector naame
ax.set_xticks(ticks)
#rotate xtick 90degrees
plt.xticks(rotation=90)
#set yticks
ax.set_yticks(ticks)
#set xlabel and ylabel to columns names
ax.set_xticklabels(dX_frame.columns)
ax.set_yticklabels(dX_frame.columns)
#plot graph
plt.show()


#%% Principal component analysis

#zero mean
df_zero_mean = np.mean(df_BR)


'''Note that this covariance function performs
the same function as the numpy function np.cov(df)
We are merely using the mathematical function as found
in data analysis text..'''
#take mean function and subtract from table
#Xnew = X - Xm, where Xm is the mean
#Xm = 1/n(summation(from i = 1 to n)xi)
#sigma = 1/(n-1)*Xt.X
def covariance(df, bias  = False):
  '''Argument
  :df: dataframe
  :return: mean of the dataset
  '''
  if isinstance(df, pd.DataFrame):
    df = df.iloc[:, 1:]
  else:
    df = df
  #get the zero mean f the data
  #meaning we try to center the data around zeo
  df = df - np.mean(df)
  #get the covariance matrix
  if bias:
    if isinstance(df, pd.DataFrame):
      cov = (1/len(df)-1) * np.dot(df, df.T)
    else:
      cov = (1/len(np.array(x).T)-1) * np.dot(df, df.T)
  else:
    cov = (1/len(df)) * np.dot(df, df.T)
  
  return cov

#the result is a 31 by 31 matrix
covar = covariance(df)

'''Next we calculate the eigenvalues 
of the covariance matrix..
-------------------------------
From the characteristics equation we know that

det|(sigma - lambdaI)| = 0
'''
#for this we would use the numpy library
# to save our time as well

def eigenval(cov):
  from scipy import linalg as la
  if isinstance(cov, object):
    #we convert object to floating number
    cov = cov.astype(float)
  else:
    cov = cov
  #find the eigen values and characteristic
  #eigen vectors
  eigen_values, eigen_vectors = la.eig(cov)
  return eigen_values, eigen_vectors

eigen_values, eigen_vectors = eigenval(covar)

#we get thw new Xi vector by using the 
#eigenvector corresponding to the highest eigenvalue
Xnew = np.dot(df_zero_mean.T, eigen_vectors[:, 0])

Xf = pd.DataFrame(Xnew.T)


for ii in range(len(features)):
  plt.scatter(df_BR['Device'], Xf.iloc[:, ii], s = 1)
#  plt.bar(df_BR['Device'], dX_frame.iloc[:, ii])
plt.xlabel('Device name')
plt.ylabel('Standard Scalar of the Xs => (X - Xm)/sd(X)')
plt.title('Scatter plot of Features against Device After PCA')



#%% PERFORM PCA ON THE DATASET

'''This will enable us to see the clusters of the dataset
according to the device name'''



#print(pearsonr(df_BR.iloc[:, 1:]))

features = list(df_BR.iloc[:, 1:].columns)

#seperating the features from the target
dX = df_BR.iloc[:, 1:]
dY = df_BR['Device']

#statndardize the X's
dX = StandardScaler().fit_transform(dX)
dX_frame = pd.DataFrame(dX, columns = features)

dX_frame.plot()
#get data correlation
corr = dX_frame.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(dX_frame.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(dX_frame.columns)
ax.set_yticklabels(dX_frame.columns)
plt.show()

'''Note here that the mean is centered around zero
We plot all the dataset using scatter plot function 
from matplotlib'''

for ii in range(len(features)):
  plt.scatter(df_BR['Device'], dX_frame.iloc[:, ii], s = 1)
#  plt.bar(df_BR['Device'], dX_frame.iloc[:, ii])
plt.xlabel('Device name')
plt.ylabel('Standard Scalar of the Xs => (X - Xm)/sd(X)')
plt.title('Scatter plot of Features against Device before PCA')
  
'''Project in 2D using PCA'''
#instantiate PCA
#function for performing PCA

#%% PCA using sklearn

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats.stats import pearsonr

#extract the features
features = list(df_BR.iloc[:, 1:].columns)


#seperating the features from the target
dX = df_BR.iloc[:, 1:]
dY = df_BR['Device']

#statndardize the X's
dX = StandardScaler().fit_transform(dX)
dX_frame = pd.DataFrame(dX, columns = features)


PCA_component = PCA(n_components = 2)

#fit data to principal component
principal_cpm = PCA_component.fit_transform(dX)

#convert to dataframe
prim_dF = pd.DataFrame(principal_cpm, columns = ['components 1', 'components 2'])
#prim_dF = pd.DataFrame(principal_cpm, columns = ['components 1'])

#final dataframe
final_df = pd.concat([prim_dF, df_BR['Device']], axis = 1)


plt.scatter(final_df['components 1'], final_df['components 2'])


#%% VISUALIZE THE DATASET

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

device_target = list(df_BR['Device'])

for targets in device_target:
  inditokeep = final_df['Device'] == targets
  ax.scatter(final_df.loc[inditokeep, 'components 1'],
             final_df.loc[inditokeep, 'components 2'],
             label = device_target)
ax.legend(targets)
ax.grid()








