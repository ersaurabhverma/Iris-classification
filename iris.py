import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris = pd.read_csv('Iris.csv')
print(iris.head)
sns.pairplot(iris.drop(labels=['Id'], axis=1), hue='Species')
X_train, X_test, y_train, y_test = train_test_split(iris[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']],iris['Species'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(pd.concat([X_test, y_test, pd.Series(y_pred, name='Predicted', index=X_test.index)],ignore_index=False, axis=1))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
