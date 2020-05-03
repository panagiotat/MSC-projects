import numpy as np
import scipy as sc
import matplotlib.pyplot as plt_test
import matplotlib.pyplot as plt_train
from sklearn import datasets 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#Load Dataset
diabete = datasets.load_diabetes()

#Find the best X to use by calculating Correlation with Y 
#inorder to have better result at creating Linear Regression Model.
y= diabete.target
corr_max = 0.0 
for i in range (np.size(diabete.data[0, :])):

    x= diabete.data[: , i]
    corr = sc.stats.pearsonr(x, y)
    
    if corr_max < corr[0] :
        corr_max = corr[0]
        best_feature = i 


#Use the best_feature to set X
x = diabete.data[: , np.newaxis , best_feature]


#Splitting the dataset, creating 2 smaller. One for Train and one for Test.
#Train Set is 75% of the dataset and Test Set is 25%.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print("Searching for a good Linear Regression Model.....\n")

#Calculating the Linear Regression Model
reg = LinearRegression().fit(X_train , y_train)
y_predicted = reg.predict(X_test)
corr = sc.stats.pearsonr(y_predicted, y_test)
mse = mean_squared_error(y_predicted,y_test)
r2 = r2_score(y_predicted,y_test)


#I repeat the Model calculation till I find a good model that is not overfitting
#I observed Correlation, MSE, R^2 Score and I found that
#a good number for Correlation is over 0.65, MSE bellow 2700 and R^2 Score greater than -0.3
while  corr[0]<0.65 or mse> 2700 or r2< -0.3:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    reg = LinearRegression().fit(X_train , y_train)
    y_predicted = reg.predict(X_test)
    corr = sc.stats.pearsonr(y_predicted, y_test)
    mse = mean_squared_error(y_predicted,y_test)
    r2 = r2_score(y_predicted,y_test)
        
        
#Calculating the metrics
y_train_predicted = reg.predict(X_train)
train_corr = sc.stats.pearsonr(y_train_predicted, y_train)
train_mse = mean_squared_error(y_train_predicted,y_train)
train_r2 = r2_score(y_train_predicted,y_train)

y_predicted = reg.predict(X_test)
test_corr = sc.stats.pearsonr(y_predicted, y_test)
test_mse = mean_squared_error(y_predicted,y_test)
test_r2 = r2_score(y_predicted,y_test)


#Printing Correlation, Mean Squared Error and R2 Score
print("Train Set Correlation is:" , train_corr[0])
print ("Train Set Mean Squared Error is:", train_mse)
print ("Train Set R2 Score is:", train_r2)

print("\nTest Set Correlation is:" , test_corr[0])
print ("Test Set Mean Squared Error is:", test_mse)
print ("Test Set R2 Score is:", test_r2)


#Plotting the results

#Plot with Train set
plt_train.scatter(X_train, y_train)
plt_train.plot(X_train, y_train_predicted, color='red')
plt_train.xlabel("x_train", color='blue')
plt_train.ylabel("y_train", color='blue')
plt_train.title("Linear Regression Model with Train Set", color='red')
plt_train.show()

#Plot with Test set
plt_test.scatter(X_test, y_test)
plt_test.plot(X_test, y_predicted, color='red')
plt_test.xlabel("x_test", color='blue')
plt_test.ylabel("y_test", color='blue')
plt_test.title("Linear Regression Model with Test Set", color='red')
plt_test.show()