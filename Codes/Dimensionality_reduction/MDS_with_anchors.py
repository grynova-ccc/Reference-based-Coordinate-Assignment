#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Code used to perform the MDS with anchors
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize

df=pd.DataFrame(np.load('INPUT_PATH'))
X = df.values

D = pairwise_distances(X, n_jobs=-1) # Precompute full pairwise distance matrix

def stress(Y_var, Y_init, D, is_anchor, n_dim): # Stress function to minimize for anchored MDS
    Y_full = Y_init.copy()
    non_anchor_idx = np.where(~is_anchor)[0]
    Y_full[non_anchor_idx] = Y_var.reshape(-1, n_dim)
    dist = pairwise_distances(Y_full)
    return np.sum((D - dist) ** 2)

def anchored_mds(X, D, n_anchors=20, n_dim=20, random_state=0): # Anchored MDS Function
    n_samples = X.shape[0]

    kmeans = KMeans(n_clusters=n_anchors, random_state=random_state).fit(X) # KMeans to find clusters
    anchor_labels = kmeans.labels_

    anchors_idx = [] # Pick actual points closest to each cluster center
    for i in range(n_anchors):
        cluster_points = np.where(anchor_labels == i)[0]
        center = kmeans.cluster_centers_[i]
        closest = cluster_points[np.argmin(np.linalg.norm(X[cluster_points] - center, axis=1))]
        anchors_idx.append(closest)
    anchors_idx = np.array(anchors_idx)
    is_anchor = np.zeros(n_samples, dtype=bool)
    is_anchor[anchors_idx] = True

    Y_init = np.random.rand(n_samples, n_dim) # Initialize coordinates

    res = minimize( # Optimize positions of non-anchor points
        stress,
        Y_init[~is_anchor].flatten(),
        args=(Y_init, D, is_anchor, n_dim),
        method="L-BFGS-B"
    )
    Y_init[~is_anchor] = res.x.reshape(-1, n_dim)

    return Y_init, anchors_idx

Y, anchors_idx = anchored_mds(X, D, n_anchors=20, n_dim=20)

reduced_df = pd.DataFrame(Y, index=df.index, columns=[f"dim{i+1}" for i in range(20)])
reduced_df.to_csv('OUTPUT_PATH')

