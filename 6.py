import pandas as pd

msg=pd.read_csv('textdocument_dataset.csv',names=['message','label'])

msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
Y=msg.labelnum

print('\nThe message and its label of first 5 instances are listed below')
X5, Y5 = X[0:5], msg.label[0:5]
for x, y in zip(X5,Y5):
    print(x,',',y)


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)
print('\nDataset is split into Training and Testing samples')
print('Total training instances :', xtrain.shape[0])
print('Total testing instances :', xtest.shape[0])


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain) #Sparse matrix
xtest_dtm = count_vect.transform(xtest)
print('\nTotal features extracted using CountVectorizer:',xtrain_dtm.shape[1])

print('\nFeatures for first 5 training instances are listed below')
df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names_out())
print(df[0:5])#tabular representation

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)

print('\nClassstification results of testing samples are given below')
for doc, p in zip(xtest, predicted):
    if p==1:
        pred = 'pos'
    else:
        'neg'
    print('%s -> %s ' % (doc, pred))

#printing accuracy metrics
from sklearn.metrics import confusion_matrix, classification_report

print('Confusion matrix')
print(confusion_matrix(ytest,predicted))
print('Classification report')
print(classification_report(ytest,predicted))

#2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
msg=pd.read_csv('naive.csv',header=None,names=['message','label'])
print("The dimensions of the dataset",msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
x=msg.message
y=msg.labelnum
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=1)
count_vect=CountVectorizer()
xtrain_dtm=count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)
clf=MultinomialNB().fit(xtrain_dtm,ytrain)
predicted=clf.predict(xtest_dtm)
print("Accuracy metrics:")
print("Accuracy of the classifier
is",metrics.accuracy_score(ytest,predicted))
print("Confusion matrix:")
print(metrics.confusion_matrix(ytest,predicted))
print("Recall and Precision:")
print(metrics.recall_score(ytest,predicted))
print(metrics.precision_score(ytest,predicted))