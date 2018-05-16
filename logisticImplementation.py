
import pandas as pd
#for logisctic regression
#let us consider y with two labels 0 &1
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pylab as pl
from numpy import asarray
#cost function is ylog(h(x))-(1-y)log(1-h(x))
import numpy as np
def calculateCostFunction(theta,b,featureX,y):
	
	totalCost=0
	allCost=[]
	iteration=[]

	finalResult=np.dot(featureX,np.transpose(theta))

	Y=y.T
	totalCost=-np.sum(Y*np.log(sigmoidFunct(finalResult+b))-(1-Y)*np.log(1-sigmoidFunct(finalResult+b)))
	
	return totalCost/len(featureX)

def sigmoidFunct(finalResult):

	
	return (np.exp(finalResult))/(1+np.exp(finalResult))

def minnTheta(theta,b,n,featureX,y):

	allCost=[]
	iterations=[]
	condition=True

	for i in range(n):

		theta=theta-0.033*partialDifTheta(theta,b,featureX,y)
		b=b-0.033*partialDifB(theta,b,featureX,y)

	#print(calculateCostFunction(theta,featureX,y)
		allCost.append(calculateCostFunction(theta,b,featureX,y))
		iterations.append(i)

	plt.plot(allCost,iterations)
	plt.show()
	return theta,b


def partialDifB(theta,b,featureX,y):
	sumMe=0
	finalResult=sigmoidFunct(np.dot(featureX,np.transpose(theta))+b)

	sumMe=(np.sum(finalResult-y.T))

	return sumMe.T/(len(featureX))
			
def partialDifTheta(theta,b,featureX,y):
	sumMe=0
	finalResult=sigmoidFunct(np.dot(featureX,np.transpose(theta))+b)

	sumMe=(np.dot(featureX.T, (finalResult-y.T).T))

	return sumMe.T/(len(featureX))

def pred(finalTheta,finalB,X_test):

	finalResult=np.dot(X_test,np.transpose(finalTheta))
	print (finalResult.shape)

	y_pred=sigmoidFunct(finalResult+finalB)
	y_new=[]
	for i in y_pred:
		if i<=0.5:
			y_new.append(0)
		else:
			y_new.append(1)

	return (y_new)

#main function start
redWine=pd.read_csv('winequality-red.csv',sep=';')
wine=pd.read_csv('winequality-white.csv',sep=';')

redWine['type']=np.array([0 for i in range(len(redWine))])
x_red=np.asarray(redWine.iloc[:,:12])
y_red=np.array([0 for i in range(len(redWine))])

wine['type']=np.array([1 for i in range(len(wine))])
y_white=np.array([0 for i in range(len(wine))])
x_white=np.asarray(wine.iloc[:,:12])

print (x_white,x_red)

wine=wine.append(redWine)

wine = shuffle(wine)

length=len(wine.columns)
ToBeScaled=wine.iloc[:,:length-1]

#plt.plot(x_red,y_red,'*')
#plt.plot(x_white,y_white,'+')
#plt.axis('equal')

#plt.show()
featureX=preprocessing.scale(ToBeScaled)

y=np.array(wine['type'])

X_train, X_test, y_train, y_test = train_test_split(featureX, y, test_size=0.33, random_state=42)

firstRandomTheta=[0 for i in range(len(wine.columns)-1)]
b=0



final=minnTheta(firstRandomTheta,b,5000,X_train,y_train)

finalTheta=final[0]

finalB=final[1]

y_pred=pred(finalTheta,finalB,X_test)

print (mean_squared_error(y_test,y_pred))


from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()



clf.fit(X_train, y_train)



pred = clf.predict(X_test)

print (mean_squared_error(y_test,pred))





