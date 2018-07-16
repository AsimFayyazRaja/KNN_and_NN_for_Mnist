import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing, cross_validation
from tqdm import tqdm
import csv

df=pd.read_csv('trainData.csv')       #getting file

df.fillna(' ', inplace=True)     #all NAN are replaced by ' '

#print(df.head())

X=[]
print("Getting features..")
with tqdm(total=len(list(df.iterrows()))) as pbar:
    for i in range(len(list(df.iterrows()))):
        X.append(df.loc[i])
        pbar.update(1)


#print(len(X))

df1=pd.read_csv('trainLabels.csv')

df1.fillna(' ',inplace=True)

y=[]
i=0
print("Getting labels..")
with tqdm(total=len(list(df1.iterrows()))) as pbar:
    for i in range(len(list(df1.iterrows()))):
        y.append(df1.loc[i])
        pbar.update(1)

#print(y[0])
#print(len(y))

#X= preprocessing.scale(X)
#X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2)       #testing upto 20%

clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(X,y)

#accuracy=clf.score(X_test, y_test)       #test on different data
#print("Accuracy: ",accuracy)

df2=pd.read_csv('testData.csv')

print("Getting features of test data..")
testX=[]
with tqdm(total=len(list(df2.iterrows()))) as pbar:
    for i in range(len(list(df2.iterrows()))):
        testX.append(df2.loc[i])
        pbar.update(1)

print("Predicting on test data..")
predictions=[]
with tqdm(total=len(testX)) as pbar:
    for i in range(len(testX)):
        x=[testX[i]]
        pred=clf.predict(x)
        predictions.append([i+1,np.take(pred,0)])
        pbar.update(1)

print("take: ",np.take(pred,0))
print(len(predictions))

csvfile = "KNN_results.csv"

i=0
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    if(i==0):
        writer.writerow(["ID","label"])
    i+=1
    writer.writerows(predictions)

print("data dumped to file")


