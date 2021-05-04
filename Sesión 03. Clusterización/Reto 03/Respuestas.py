# -*- coding: utf-8 -*-
"""
Jaffet León Chávez

Bienvdenido a este último reto de la sesión 03. Vamos a aplicar todo lo visto en la
sesión 03 de K-means clustering
"""

# 1. Use las siguientes librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

os.chdir("C:/Users/Pelu/OneDrive - UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO/cosas de Jaff Koalita/Bedu/ML/Sesión 03/Ejercicio por Jaffet León")

#2 Use pandas para leer el archivo CC GENERAL CSV y use pandas.fillna con inplace
# con un boolean para decirle que queremos fuera a los NA

raw_df = pd.read_csv('CC GENERAL.csv')
raw_df = raw_df.drop('CUST_ID', axis = 1) 
raw_df.fillna(method ='ffill', inplace = True) 
raw_df.head(2)

# 3. Estandarice, normalice

scaler = StandardScaler() 
scaled_df = scaler.fit_transform(raw_df) 

normalized_df = normalize(scaled_df)

# 4. Convierta el array de numpy a un data frame de pandas

normalized_df = pd.DataFrame(normalized_df) 

# 5.  Reduzca las dimensiones, variables que ustedes crean pertinente

pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(normalized_df) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 
  
X_principal.head(2)

# 6. Aplique el criterio del 'codo' que acabamos de aprender en clase

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X_principal)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

# 7. Haga el método de K-mean clustering y grafique. Como bonus no obligatorio
# grafique los centroides

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_principal)

plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = KMeans(n_clusters = 3).fit_predict(X_principal), cmap =plt.cm.winter) 
plt.show() 


# Con Centroides

plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = KMeans(n_clusters = 3).fit_predict(X_principal), cmap =plt.cm.winter) 
plt.plot(X_principal['P1'], X_principal['P2'], 'k.', markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='o', s=10, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the credit card fraud dataset (PCA-reduced data)\n'
          'Centroids are marked with white circle')
plt.show()

