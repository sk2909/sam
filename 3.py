import numpy as np
import pandas as pd

PlayTennis = pd.read_csv("tennis.csv")

from sklearn.preprocessing import LabelEncoder

Le = LabelEncoder()
PlayTennis['outlook'] = Le.fit_transform(PlayTennis['outlook'])
PlayTennis['temp'] = Le.fit_transform(PlayTennis['temp'])
PlayTennis['humidity'] = Le.fit_transform(PlayTennis['humidity'])
PlayTennis['play'] = Le.fit_transform(PlayTennis['play'])
print(PlayTennis)

y=PlayTennis['play']
x=PlayTennis.drop(['play'],axis=1)

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf=clf.fit(x,y)

X_pred = clf.predict(x)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y,X_pred))
print(classification_report(y,X_pred))
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.figure()
tree.plot_tree(clf,fontsize=6)
plt.savefig('tree.jpg', format='jpg',bbox_inches='tight')
plt.show()