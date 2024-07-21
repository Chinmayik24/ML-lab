#pca

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnpca
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.datasets import load_iris

X=load_iris().data
y=load_iris().target

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

cor_coef=np.corrcoef(X_scaled.T)
sns.heatmap(cor_coef,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Heatmap after standardization')
plt.show()

pca=sklearnpca(n_components=2)
X_project=pca.fit_transform(X_scaled)

print('Shape of data: ',X.shape)
print('Shape of projected data: ',X_project.shape)

pc1=X_project[:,0]
pc2=X_project[:,1]

plt.scatter(pc1,pc2,c=y,cmap='jet')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()

#lda

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sklearnlda
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.datasets import load_iris

X=load_iris().data
y=load_iris().target

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

cor_coef=np.corrcoef(X_scaled.T)
sns.heatmap(cor_coef,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Heatmap after standardization')
plt.show()

lda=sklearnlda(n_components=2)
X_project=lda.fit_transform(X_scaled,y)

print('Shape of data: ',X.shape)
print('Shape of projected data: ',X_project.shape)

ld1=X_project[:,0]
ld2=X_project[:,1]

plt.scatter(ld1,ld2,c=y,cmap='jet')
plt.xlabel('ld1')
plt.ylabel('ld2')
plt.show()
