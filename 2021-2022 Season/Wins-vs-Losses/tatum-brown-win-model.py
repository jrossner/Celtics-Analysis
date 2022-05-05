tatumScor = [20,18,31,41,23,27,20,14,10,32,22,27,21,23,34,37,33,30,15,24,8,26,37,31,34,29,24,42,27,25,17,18,25,'I','I','I','I',19,36,19,24,33,20,23,27,12,27,51,36,20,38,20,19,24,15,19,24,38,28,22,30,26,24,33,37,54,44,31,21,26,32,30,36,26,34,'I',23,31,22,16,'I',31]
brownScor = [46,9,'I',30,13,34,28,28,17,'I','I','I','I','I','I','I','I',19,13,16,16,9,'I','I','I','I','I',19,20,23,30,34,25,26,30,24,50,30,16,22,26,34,21,19,23,21,22,18,30,26,31,29,15,13,26,22,12,17,29,31,18,27,23,2,'I',21,15,22,14,26,30,30,25,26,31,'I',28,32,32,25,22,18]
tatumPass = [4,4,2,8,2,3,3,3,2,2,7,1,2,5,5,2,5,3,3,2,10,2,5,4,3,2,3,4,6,2,6,5,4,'I','I','I','I',3,9,1,1,4,5,4,3,3,7,7,6,4,7,5,9,1,7,4,5,3,6,5,4,6,3,7,5,3,3,6,4,4,4,7,6,3,6,'I',6,6,7,8,'I',3]
brownPass = [6,0,'I',3,1,2,3,3,2,'I','I','I','I','I','I','I','I',0,0,3,3,3,'I','I','I','I','I',5,2,5,4,3,3,4,0,3,4,1,1,11,6,3,1,4,2,6,5,1,3,3,3,2,6,2,3,9,4,3,3,6,6,0,8,1,'I',5,5,4,4,3,0,0,1,5,2,'I',6,7,5,4,11,4]
gameResults = ['L','L','W','W','L','L','L','W','W','L','W','W','L','W','L','W','W','W','L','L','W','W','L','W','L','L','L','W','L','W','L','W','L','L','L','W','W','L','L','W','W','W','L','W','W','L','L','W','W','L','W','W','W','W','W','W','W','W','W','L','W','W','L','W','W','W','W','W','L','W','W','W','W','W','W','L','L','W','W','W','L','W']

len(tatumScor)
len(brownScor)
len(brownPass)
len(tatumPass)
len(gameResults)

data = []
for i in range(len(gameResults)):
    if (tatumScor[i] != 'I' and brownScor[i] != 'I'):
        data.append([gameResults[i],tatumScor[i],tatumPass[i],brownScor[i],brownPass[i]])

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.DataFrame(data,columns=['result','tatum-points','tatum-assists','brown-points','brown-assists'])
df.head()

X = df[['tatum-points','tatum-assists','brown-points','brown-assists']]
Y = df['result']

trainX,testX,trainY,testY = train_test_split(X,Y, test_size = .333)
model = LogisticRegression()
model.fit(trainX,trainY)

model.coef_
model.intercept_
model.classes_

pred = model.predict(testX)
mat = metrics.confusion_matrix(testY,pred)
cm = metrics.ConfusionMatrixDisplay(mat,['Loss','Win'])
cm.plot()

pp = model.predict_proba(X)[::,1]
fpr,tpr,_ = metrics.roc_curve(Y,pp,pos_label='W')
auc = metrics.roc_auc_score(Y,pp)
plt.plot(fpr,tpr)

#whole dataset
mod = LogisticRegression()
mod.fit(X,Y)

mod.coef_
mod.intercept_
mod.classes_

pred = mod.predict(X)
mat = metrics.confusion_matrix(Y,pred)
cm = metrics.ConfusionMatrixDisplay(mat,['Loss','Win'])
cm.plot()

mod.coef_
labels = ['tatum-points','tatum-assists','brown-points','brown-assists']
for i in range(len(mod.coef_[0])):
    print(f'Each {labels[i]} is associated with a change in odds of a win by {exp(mod.coef_[0][i])}')
