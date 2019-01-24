#Polynomial Regression_sklearn.py
#implementing polynomial regression using sklearn libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4,random_state=0)
len_train = len(x_train)
len_test = len(x_test)
Loss_train_lib = np.zeros(10)
Loss_test_lib = np.zeros(10)
for i in range(1,10): # fit regression for polynomial of degree i 
    poly = PolynomialFeatures(i)
    x_train_poly = poly.fit_transform(x_train.reshape(len_train,1))
    reg = LinearRegression().fit(x_train_poly,y_train.reshape(len_train,1))
    Loss_train_lib[i] = mean_squared_error(y_train,reg.predict(x_train_poly))
    print("MSE/Loss on train data for {} degree polynomial is {}".format(i,Loss_train_lib[i]))
#test data calculations
    x_test_poly = poly.fit_transform(x_test.reshape(len_test,1))
    Loss_test_lib[i] = mean_squared_error(y_test,reg.predict(x_test_poly))
    print("MSE/Loss on test data for {} degree polynomial is {}".format(i,Loss_test_lib[i]))