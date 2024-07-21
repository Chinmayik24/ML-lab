import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix

def euc_dis(x1,x2):
    dis=np.sqrt(np.sum((x1-x2)**2)) #np.sum(np.abs(x1-x2))
    return dis

class KNN:
    def __init__(self,k):
        self.k=k

    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    def predict(self,X):
        predictions=[self._predict(x) for x in X]
        return predictions

    def _predict(self,x):
        distance=[euc_dis(x,x_train) for x_train in self.X_train]
        k_indices=np.argsort(distance)[:self.k]
        k_labels=[self.y_train[i] for i in k_indices]
        most_common=Counter(k_labels).most_common()
        return most_common[0][0]

data=pd.read_csv('glass.csv')
y=data['Type'].values
X=data.drop('Type', axis=1).values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

cl=KNN(k=3)
cl.fit(X_train,y_train)

predictions=cl.predict(X_test)

print(predictions)

acc=np.sum(predictions==y_test)/len(y_test)
print(acc)

cor_coef=np.corrcoef(data,rowvar=False)
sns.heatmap(cor_coef,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('heatmap')
plt.show()

confu_mat=confusion_matrix(y_test,predictions)
sns.heatmap(confu_mat,annot=True,cmap='Blues')
plt.title('confusion matrix')
plt.show()
