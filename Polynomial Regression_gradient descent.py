#Polynomial Regression_gradient descent.py
#implementing polynomial regression using gradient descent without using linear regression libraries from sklearn
import numpy as np
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4,random_state=0)
lr_rate = 0.00001 #learning rate 
Loss_train = np.zeros(10)
Loss_test = np.zeros(10)
for i in range(1,10):
    len_train = len(x_train)
    x0 = np.ones(shape=(len_train,1))
    w = np.random.rand(i+1,1) #(i+1)x1
    phi_x = x0.reshape(len_train,1) #6000x1
    for j in range(i):
        phi_x = np.concatenate((phi_x,np.power(x_train.reshape(len_train,1),j+1)),axis=1) #6000x(i+1)
    iterations = 0
#     print(phi_x[1:10])
    while(iterations<50000):
        A = (y_train.reshape(len_train,1)-np.dot(phi_x,w)) # 6000x1
        Loss = (np.dot(A.transpose(),A)) # 1x1
        w = w - lr_rate*(-2*((np.dot(A.transpose(),phi_x)).transpose()))
        iterations+=1
        if(abs(Loss_train[i]-Loss)<0.0000001):
            Loss_train[i] = Loss
            break
        else:
            Loss_train[i] = Loss
    print("MSE/loss for {} degree polynomial on train data is {}".format(i, Loss_train[i]))
# test loss calculation 
    len_test = len(x_test)
    x0_test = np.ones(len_test)
    phi_x_test = x0_test.reshape(len_test,1)
    for j in range(i):
        phi_x_test = np.concatenate((phi_x_test,np.power(x_test.reshape(len_test,1),j+1)),axis=1)
    A_test = (y_test.reshape(len_test,1)-np.dot(phi_x_test,w.reshape(i+1,1)))
    Loss_test[i] = (np.dot(A_test.transpose(),A_test))
    print("MSE/loss for {} degree polynomial on test data is {}".format(i, Loss_test[i]))