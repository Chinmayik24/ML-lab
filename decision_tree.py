import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('weather.csv')
df.head()

for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    
sns.countplot(x = 'play',data=df)
plt.title('playing ')
plt.show()

sns.countplot(x='play',hue='humidity',data=df)
plt.title('HUMIDITY VS PLAY')
plt.show()

sns.countplot(x='play',hue='temperature',data=df)
plt.title('HUMIDITY VS PLAY')
plt.show()

sns.countplot(x='play',hue='windy',data=df)
plt.title('HUMIDITY VS PLAY')
plt.show()

sns.countplot(x='play',hue='outlook',data=df)
plt.title('HUMIDITY VS PLAY')
plt.show()

X = df.drop('play', axis=1)
y = df['play']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

plot_tree(clf, feature_names=X.columns, filled=True)
plt.show()
