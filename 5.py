from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
dataset=pd.read_csv('diabetes.csv')
X=np.array(dataset.iloc[:,0:-1])
Y=np.array(dataset.iloc[:,-1])
dataset=datasets.load_iris()
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=1)
model=GaussianNB()
model.fit(xtrain,ytrain)
predicted=model.predict(xtest)
print(metrics.confusion_matrix(ytest,predicted))
print(metrics.accuracy_score(ytest,predicted))