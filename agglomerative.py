import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage

iris=load_iris()
data=iris.data[:6]

def proximity_matrix(data):
    n=data.shape[0]
    proximity_matrix=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            proximity_matrix[i,j]=np.linalg.norm(data[i]-data[j])
            proximity_matrix[j,i]=proximity_matrix[i,j]
    return proximity_matrix

print("Proximity matrix:")
print(proximity_matrix(data))

def dendro_plot(data,method):
    linkage_matrix=linkage(data,method=method)
    dendrogram(linkage_matrix)
    plt.title(f"dendrogram for {method} linkage")
    plt.xlabel("data points")
    plt.ylabel("values")
    plt.show()

print(dendro_plot(data,'single'))
print(dendro_plot(data,'complete'))
