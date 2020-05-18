
import numpy as np
import pickle 
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

pickle_in=open("data.pkl","rb")
indata=[]
indata=pickle.load(pickle_in)
xval=[]
xval=indata[:,0:1]
yval=[]
yval=indata[:,1:2]

X_train, X_test, y_train, y_test = train_test_split(xval, yval, test_size=0.1,random_state=1)

trainsetx=[]
trainsety=[]
trainsetx=np.array_split(X_train,10)
trainsety=np.array_split(y_train,10)

avgbias=[]
avgvar=[]
for i in range(1,10):
    expected=[]
    expected=np.zeros(len(X_test))
    globaly=[]
#     print(len(expected))
    for j in range(10):
        xin=trainsetx[j]
        yout=trainsety[j]
        polyn=PolynomialFeatures(degree=i)
        x_poly_train=polyn.fit_transform(xin)
        reg=LinearRegression()
        reg.fit(x_poly_train,yout)
        x_poly_test=polyn.fit_transform(X_test)
        y_pred=reg.predict(x_poly_test)
        globaly.extend(y_pred)
        for k in range(len(expected)):
            expected[k]+=y_pred[k]
        
    expected=np.array(expected)/10
    bias=[]
    bias=np.zeros(len(X_test))
    for j in range(len(bias)):
        bias[j]=expected[j]-y_test[j]
        bias[j]=bias[j]**2
    avgbias.append(np.mean(bias))    
    var=[]
    var=np.zeros(len(globaly))
    summ=0
    for j in range(len(globaly)):
        var[j]=globaly[j]-expected[(j%(len(X_test)))]
        var[j]=var[j]**2
        summ+=var[j]
    avgvar.append((summ/len(X_test)))
    
xx=np.arange(1,10)
# print(avgbias)
# print(avgvar)
plt.plot(xx,avgvar,marker='.',color='red',label='variance')
plt.plot(xx,avgbias,marker='.',color='blue',label='bias')
plt.xlabel('complexity of polynomial')
plt.ylabel('error')

# plt.plot(xx,avgvar,marker='.',color='red')
# plt.plot(xx,avgbias,marker=',',color='blue')
plt.show()
