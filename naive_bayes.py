import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('TitanicDataset.csv')

df.head()
df.info()
df.describe()
df.columns
df.isnull().sum()

df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df.drop(columns=['Cabin'],inplace=True)

df['Sex']=df['Sex'].map({'male':0,'female':1})
df=pd.get_dummies(df,columns=['Embarked'],drop_first=True)

sns.countplot(x='Survived',data=df)
plt.title('Survived vs Not Survived')
plt.show()

sns.countplot(x='Survived',hue='Sex',data=df)
plt.title('Survival by Sex')
plt.show()

sns.histplot(df['Age'],bins=30,kde=True)
plt.title('Age histogram')
plt.show()

sns.countplot(x='Survived',hue='Pclass',data=df)
plt.title('Survival by Pclass')
plt.show()

X=df.drop(columns=['Name','PassengerId','Ticket','Survived'])
y=df['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

nb_model=GaussianNB()
nb_model.fit(X_train,y_train)

predictions=nb_model.predict(X_test)
acc=accuracy_score(y_test,predictions)

con_mat=confusion_matrix(y_test,predictions)
sns.heatmap(con_mat,annot=True,cmap='coolwarm')
plt.show()

print(classification_report(y_test,predictions))
print('Accuracy: ',acc)
