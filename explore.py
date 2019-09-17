# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:28:44 2019

@author: shrey
"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv('train_values.csv')
labels = pd.read_csv('train_labels.csv')

#%%
#for missing values
isnull = data.isnull()
for i in range(isnull.shape[0]):
    for j in range(isnull.shape[1]):
        if isnull.iloc[i,j] == True:
            print(i,j)
            

#%%
#n_subjects = np.unique(data.patient_id.values)
#data = data.drop(columns = ['patient_id'])

thal_values = np.unique(data.thal.values)
data['thal']=data['thal'].astype('category',thal_values).cat.codes


sex = np.unique(data.sex.values)
data['sex']=data['sex'].astype('category',sex).cat.codes
#%%

standard_scaler = preprocessing.StandardScaler()
data_scaled = standard_scaler.fit_transform(data.loc[:, data.columns != 'patient_id'])
data_scaled = pd.DataFrame(data_scaled, columns = data.columns.values[data.columns.values != 'patient_id'])
#%%
data_scaled['patient_id'] = data.patient_id

#%%

tsne = TSNE(n_components=2, random_state=0)
results_2d = tsne.fit_transform(data_scaled.loc[:, data_scaled.columns != 'patient_id'])

all_data = pd.merge(data_scaled, labels, on = 'patient_id')
#results_2d = tsne.fit_transform(all_data.loc[:, all_data.columns != 'patient_id'])

all_data['tsne_2d_one'] = results_2d[:,0]
all_data['tsne_2d_two'] = results_2d[:,1]

plt.figure()
sns.scatterplot(
    x="tsne_2d_one", y="tsne_2d_two",
    hue="heart_disease_present", style = "heart_disease_present",
    palette=sns.color_palette("husl", 2),
    data=all_data,
    legend="full",
    alpha=1, 
    s = 100
)

#%%

pca = PCA(n_components=3)
pca_result = pca.fit_transform(data)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
Explained variation per principal component: [0.09746116 0.07155445 0.06149531]