import pandas as pd 
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
import numpy as np
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler , OneHotEncoder
from sklearn.svm import SVR 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
from matplotlib.pyplot import figure
from sklearn.metrics import explained_variance_score , mean_squared_error, mean_absolute_error , r2_score





#SVSearch is a greedy function that was made to find the best parameters for SVRegression 
def SVSearch(train_x ,  test_x , train_y ,test_y):
    
    #best contains the best parameters for my model
    best=[]
    #gamma_data contains the gamma values that I tried
    gamma_data= []
    #gamma_data_score contains the score of the model in test set for different values of gamma
    gamma_data_score = []
    #gamma_data_train_score contains the score of the model in train set for different values of gamma
    gamma_data_train_score = []    
    #C_data contains the C values that I tried
    C_data = []
    #C_data_score contains the score of the model in test set for different values of C
    C_data_score = []
    #C_data_train_score contains the score of the model in train set for different values of C
    C_data_train_score = []

    #epsilon_data contains the epsilon values that I tried
    epsilon_data = []
    #epsilon_data_score contains the score of the model in test set for different values of epsilon
    epsilon_data_score = []
    #epsilon_data_train_score contains the score of the model in train set for different values of epsilon
    epsilon_data_train_score = []
    
    #Here is the greedy search. I use three whiles (one for every parameter). I choose the range and the step of every try. 
    #clf = SVR(gamma=0.01, C=1, epsilon=0.01 , degree =10, kernel='rbf')
    clf = SVR(gamma=0.01, C=1, epsilon=0.01 , degree =30, kernel='linear')
    clf.fit( train_x , train_y) 
    y_predict = clf.predict( test_x)
    best_mean_absolute_error = mean_absolute_error(y_predict , test_y)
    best_epsilon = 0.01
    i=0.01
    
    while i<=30:
        
        clf = SVR(gamma=0.01, C=1, epsilon=i , degree =10, kernel='rbf')
        clf.fit( train_x , train_y) 
        y_predict_train = clf.predict( train_x)
        y_predict = clf.predict( test_x)
        t1 = mean_absolute_error(y_predict_train , train_y)
        t2 = mean_absolute_error(y_predict , test_y)
        if t2<best_mean_absolute_error :
            best_mean_absolute_error=t2
            best_epsilon = i            
        
        epsilon_data.append(i)
        epsilon_data_train_score.append(t1)
        epsilon_data_score.append(t2)
        
        #print(i , len(clf.support_vectors_ ),t1 ,t2 )
        if i<=10:
            
            i=i+0.01
        else:
            i=i+1

    
    best_gamma = 0.01
    i=0.01    
    while i<=3:                    
            
        clf = SVR(gamma=i, C=1, epsilon=best_epsilon , degree =10, kernel='rbf')
        clf.fit( train_x , train_y) 
        y_predict_train = clf.predict( train_x)
        y_predict = clf.predict( test_x)
        t1 = mean_absolute_error(y_predict_train , train_y)
        t2 = mean_absolute_error(y_predict , test_y)
        if t2<best_mean_absolute_error :
            best_mean_absolute_error=t2
            best_gamma = i
        
        gamma_data.append(i)
        gamma_data_train_score.append(t1)
        gamma_data_score.append(t2)
                
        #print(i , len(clf.support_vectors_ ),t1 ,t2 )
        i=i+0.01  
    

    best_C = 1
    i=1
    while i<=6000:
             
        clf = SVR(gamma=best_gamma, C=i, epsilon=best_epsilon , degree =10, kernel='rbf')
        clf.fit( train_x , train_y) 
        y_predict_train = clf.predict( train_x)
        y_predict = clf.predict( test_x)
        t1 = mean_absolute_error(y_predict_train , train_y)
        t2 = mean_absolute_error(y_predict , test_y)
        if t2<best_mean_absolute_error :
            best_mean_absolute_error=t2
            best_C = i
        
        C_data.append(i)
        C_data_train_score.append(t1)
        C_data_score.append(t2)
        #print(i , len(clf.support_vectors_ ),t1 ,t2 )
        
        if i<50:
            i=i+1
        
        elif i>=50 and i<100:
            i = i+2 
        elif i>=100 and i<500:
            i = i+5
        elif i>=500 and i<2000:
            i=i+10
        elif i>=2000:
            i=i+100  


    best.append([best_epsilon , best_gamma , best_C  ])
    
    #I use plots function inorder to print the diagrams of my data
    plots(gamma_data, gamma_data_train_score, gamma_data_score, C_data , C_data_train_score, C_data_score, epsilon_data ,epsilon_data_train_score,epsilon_data_score)
    
    return best
    

#In this function I preprocess the data. So I prepare and transform them into an input for my model.
def data_preprocessing1(dataset, test_dataset, pairs ):
    
    data = []
    test_data = []
    
    for line in pairs:
        for attribute in line[0]:      
            for method in line[1]:
                if method==enc:
                                        
                    temp_dataset = dataset.append(test_dataset)
                    method.fit(temp_dataset[attribute].values.reshape(-1, 1))
                    pre_data = method.transform(dataset[attribute].values.reshape(-1, 1)).toarray()
                    test_pre_data = method.transform(test_dataset[attribute].values.reshape(-1, 1)).toarray()
                    if len(data)==0:
                        data = pre_data
                        test_data = test_pre_data
                    else:
                        data = np.append(data, pre_data , axis=1)
                        test_data = np.append(test_data, test_pre_data , axis=1)
                else:
                    #I fit the Scaler into Train dataset and after that I tranform both the Train and the Test set with that Scaler
                    method.fit(dataset[attribute].values.reshape(-1, 1))
                    pre_data = method.transform(dataset[attribute].values.reshape(-1, 1))
                    test_pre_data = method.transform(test_dataset[attribute].values.reshape(-1, 1))
                    if len(data)==0:
                        data = pre_data
                        test_data = test_pre_data
                    else:
                        data = np.append(data, pre_data , axis=1)
                        test_data = np.append(test_data, test_pre_data , axis=1)    
    return data , test_data


#In the function plots I plot the data into diagrams
def plots (gamma_data, gamma_data_train_score, gamma_data_score, C_data , C_data_train_score, C_data_score, epsilon_data ,epsilon_data_train_score,epsilon_data_score):
    
    figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
    plt2.plot(epsilon_data, epsilon_data_train_score , color = "red")
    plt2.plot(epsilon_data, epsilon_data_score , color="blue")
    plt2.ylabel('Absolute Error score')
    plt2.xlabel('Epsilon value')
    plt2.show()
    
    figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(gamma_data, gamma_data_train_score , color = "red" )
    plt.plot(gamma_data, gamma_data_score , color="blue")
    plt.ylabel('Absolute Error score')
    plt.xlabel('Gamma value')
    plt.show()
    
    figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
    plt1.plot(C_data, C_data_train_score , color = "red")
    plt1.plot(C_data, C_data_score , color="blue")
    plt1.ylabel('Absolute Error score')
    plt1.xlabel('C value')
    plt1.show()
    






#Load the three datasets and shuffle them
train_set = pd.read_csv("AB_NYC_2019.csv")
train_set = train_set.fillna(0)
train_set = train_set.sample(frac=1)

train_set1 = pd.read_csv("googleplaystore.csv") 
train_set1 = train_set1.fillna(0)
train_set1 = train_set1.sample(frac=1)

train_set2 = pd.read_csv("StudentsPerformance.csv") 
train_set2 = train_set2.fillna(0)
train_set2 = train_set2.sample(frac=1)


#Parameter a is how many data I want to use, inorder to train and test my model. I choose the target and seperate it from the input
a=500
train_set = train_set[0:a]
x_train = train_set.drop(columns="price")
y= train_set.values[:,9]

train_set1 = train_set1[0:a]
x_train1 = train_set1.drop(columns="Rating")
y1= train_set1.values[:,2]

train_set2 = train_set2[0:a]
x_train2 = train_set2.drop(columns="math score")
y2= train_set2.values[:,5]

#Split the data into Train and Test set
X_train, X_test, y_train, y_test = train_test_split(train_set , y, test_size=0.4)
X_train1, X_test1, y_train1, y_test1 = train_test_split(train_set1 , y1, test_size=0.4)
X_train2, X_test2, y_train2, y_test2 = train_test_split(train_set2 , y2, test_size=0.4)

#Setting up the pre_processing models 
scaler = MinMaxScaler()
enc = OneHotEncoder( sparse="True"  )

#Preprocess the data
X_train , X_test = data_preprocessing1(dataset=X_train , test_dataset= X_test,  pairs= [ [["latitude","longitude" , "minimum_nights", "number_of_reviews" , "reviews_per_month", "calculated_host_listings_count", "availability_365"],[scaler]] , [["neighbourhood_group","neighbourhood", "room_type"],[enc]]])
X_train1 , X_test1 = data_preprocessing1(dataset=X_train1 , test_dataset= X_test1, pairs= [   [["Reviews", "Price" , "Category","Installs", "Size", "Genres"],[enc]]])
X_train2 , X_test2 = data_preprocessing1(dataset=X_train2 , test_dataset= X_test2, pairs= [ [["reading score","writing score"],[scaler]],  [["gender", "race/ethnicity" , "parental level of education","lunch", "test preparation course"],[enc]]])


#Use SVSearch inorder to find the best parameters for epsilon, gamma and C. After that plot the all the tries into a diagram
Airbnb_houses = SVSearch(X_train, X_test, y_train, y_test)
Rating_playstore = SVSearch(X_train1, X_test1, y_train1, y_test1)
Math_score = SVSearch(X_train2, X_test2, y_train2, y_test2)








