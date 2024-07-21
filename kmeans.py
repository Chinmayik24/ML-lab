import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def kmeans(X,K,maxiters=100):
    centroids=X[:K]

    for _ in range(maxiters):
        expand_x=X[:,np.newaxis]
        euc_dis=np.linalg.norm(expand_x-centroids,axis=2)
        labels=np.argmin(euc_dis,axis=1)

        new_centroids=np.array([X[labels==k].mean(axis=0) for k in range(K)])

        if np.all(new_centroids==centroids):
            break

        centroids=new_centroids

    return labels,centroids

X=load_iris().data
K=3
labels,centroids=kmeans(X,K)
print("labels:",labels)
print("Centroids:",centroids)

plt.scatter(X[:,0],X[:,1],c=labels)
plt.scatter(centroids[:,0],centroids[:,1],marker='x',c='red',s=200)
plt.show()
