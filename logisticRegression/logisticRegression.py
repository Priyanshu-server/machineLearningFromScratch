import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(path):
	df = pd.read_csv(path)
	return df
def preprocess_dataset(path):
	df = load_dataset(path)
	df = pd.concat([pd.Series(1,index = df.index,name = '00'),df],axis = 1)
	return df.drop(columns = ['target']),df['target']


def h(x,theta):
	z = theta@x.T
	return 1/(1+np.exp(-(z))) - 0.0000001


def cost(x,y,theta):
	y1 = h(x, theta)
	return -(1/len(x))*np.sum(y*np.log(y1) + (1-y)*np.log(1-y1))

def gradientDescent(x,y,theta,epochs,lr = 0.01):
	m = len(x)
	J = []
	for i in range(0,epochs):
		h_x = h(x, theta)
		for i in range(0,len(x.columns)):
			theta[i] -= (lr/m)*np.sum((h_x-y)*x.iloc[:,i])
		J.append(cost(x,y,theta))
	return J,theta

def predict(x,y,theta,epochs,lr = 0.01):
	J,th = gradientDescent(x, y, theta, epochs,lr)
	h_x = h(x, theta)
	for i in range(len(h_x)):
		h_x[i] = 1 if h_x[i]>=0.5 else 0
	y = list(y)
	acc = np.sum([y[i]==h_x[i] for i in range(len(y))])/len(y)
	return J,acc






x,y = preprocess_dataset('heart.csv')
print(x.head(),y.head())

theta = [0.5]*len(x.columns)
J,acc = predict(x,y,theta,25000,0.0001)
print("\nTheta : {}\nAccuracy : {}".format(theta,acc))
plt.figure(figsize = (12, 8))
plt.scatter(range(0, len(J)), J)
plt.show()