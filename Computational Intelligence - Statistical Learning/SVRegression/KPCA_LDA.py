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
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 

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
    
    clf = SVC(gamma=0.01, C=1 , degree =30, kernel='rbf')
    clf.fit( train_x , train_y) 
    y_predict = clf.predict( test_x)
    best_accuracy_score = accuracy_score(y_predict , test_y)
    
   

    
    best_gamma = 0.01
    i=0.01    
    while i<=3:                    
            
        clf = SVC(gamma=i, C=1,  degree =10, kernel='rbf')
        clf.fit( train_x , train_y) 
        y_predict_train = clf.predict( train_x)
        y_predict = clf.predict( test_x)
        t1 = accuracy_score(y_predict_train , train_y)
        t2 = accuracy_score(y_predict , test_y)
        if t2<best_accuracy_score :
            best_accuracy_score=t2
            best_gamma = i
            
        
        gamma_data.append(i)
        gamma_data_train_score.append(t1)
        gamma_data_score.append(t2)
                
        #print(i , len(clf.support_vectors_ ),t1 ,t2 )
        i=i+0.01  
    

    best_C = 1
    i=1
    while i<=6000:
             
        clf = SVC(gamma=best_gamma, C=i , degree =10, kernel='rbf')
        clf.fit( train_x , train_y) 
        y_predict_train = clf.predict( train_x)
        y_predict = clf.predict( test_x)
        t1 = accuracy_score(y_predict_train , train_y)
        t2 = accuracy_score(y_predict , test_y)
        if t2<best_accuracy_score :
            best_accuracy_score=t2
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


    best.append([best_gamma , best_C  ])
    
    #I use plots function inorder to print the diagrams of my data
    plots(gamma_data, gamma_data_train_score, gamma_data_score, C_data , C_data_train_score, C_data_score)
    
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
def plots (gamma_data, gamma_data_train_score, gamma_data_score, C_data , C_data_train_score, C_data_score):
    

    
    figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(gamma_data, gamma_data_train_score , color = "red" )
    plt.plot(gamma_data, gamma_data_score , color="blue")
    plt.ylabel('Accuracy Score')
    plt.xlabel('Gamma value')
    plt.show()
    
    figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
    plt1.plot(C_data, C_data_train_score , color = "red")
    plt1.plot(C_data, C_data_score , color="blue")
    plt1.ylabel('Accuracy Score')
    plt1.xlabel('C value')
    plt1.show()
    

#Load the three datasets and shuffle them
train_set = pd.read_csv("2018.csv")
train_set = train_set.fillna(0)
train_set = train_set.sample(frac=1)
enc_score = KBinsDiscretizer(n_bins=3, encode='ordinal')
X_binned = enc_score.fit_transform(train_set["Score"].values.reshape(-1, 1))
train_set['Score'] = X_binned

train_set1 = pd.read_csv("diamonds.csv") 
train_set1 = train_set1.fillna(0)
train_set1 = train_set1.sample(frac=1)

train_set2 = pd.read_csv("winequality-red.csv") 
train_set2 = train_set2.fillna(0)
train_set2 = train_set2.sample(frac=1)





#Parameter a is how many data I want to use, inorder to train and test my model. I choose the target and seperate it from the input

a=500
enc_y = OneHotEncoder( sparse="True"  )
train_set = train_set[0:a]
x_train = train_set.drop(columns="Score")
y= train_set.values[:,2]
y = enc_y.fit_transform(y.reshape(-1, 1)).toarray()
y = [np.where(r==1)[0][0] for r in y]
#y = y.astype(int)

enc_y1 = OneHotEncoder( sparse="True"  )
train_set1 = train_set1[0:a]
x_train1 = train_set1.drop(columns="cut")
y1= train_set1.values[:,2]
y1 = enc_y1.fit_transform(y1.reshape(-1, 1)).toarray()
y1 = [np.where(r==1)[0][0] for r in y1]
#y1 = y1.astype(int)


enc_y2 = OneHotEncoder( sparse="True"  )
train_set2 = train_set2[0:a]
x_train2 = train_set2.drop(columns="quality")
y2= train_set2.values[:,11]
y2 = enc_y2.fit_transform(y2.reshape(-1, 1)).toarray()
y2 = [np.where(r==1)[0][0] for r in y2]
#y2 = y2.astype(int)

#Split the data into Train and Test set
X_train, X_test, y_train, y_test = train_test_split(x_train , y, test_size=0.4)
X_train1, X_test1, y_train1, y_test1 = train_test_split(x_train1 , y1, test_size=0.4)
X_train2, X_test2, y_train2, y_test2 = train_test_split(x_train2 , y2, test_size=0.4)

#Setting up the pre_processing models 
scaler = MinMaxScaler()
enc = OneHotEncoder( sparse="True"  )

#Preprocess the data
X_train , X_test = data_preprocessing1(dataset=X_train , test_dataset= X_test,  pairs= [ [["GDP per capita" , "Social support", "Healthy life expectancy" , "Freedom to make life choices", "Generosity", "Perceptions of corruption"],[scaler]] , [[],[enc]]])
X_train1 , X_test1 = data_preprocessing1(dataset=X_train1 , test_dataset= X_test1, pairs= [   [["carat", "depth" , "table","price", "x", "y","z"],[scaler]], [["color","clarity"],[enc]]])
X_train2 , X_test2 = data_preprocessing1(dataset=X_train2 , test_dataset= X_test2, pairs= [ [["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],[scaler]],  [[],[enc]]])



#kpca = KernelPCA(kernel="rbf", n_components=3)
kpca = KernelPCA(kernel="rbf", n_components=5)
#kpca = KernelPCA(kernel="rbf", n_components=6)
kpca.fit(X_train)
X_kpca = kpca.transform(X_train)
X_kpca_test = kpca.transform(X_test)
lda = LinearDiscriminantAnalysis()
lda.fit(X_kpca, y_train)
X_lda = lda.transform(X_kpca)
X_lda_test = lda.transform(X_kpca_test)


#kpca1 = KernelPCA(kernel="rbf",  n_components=5)
kpca1 = KernelPCA(kernel="rbf", n_components=14)
#kpca1 = KernelPCA(kernel="rbf", n_components=22)
kpca1.fit(X_train1)
X_kpca1 = kpca1.transform(X_train1)
X_kpca_test1 = kpca1.transform(X_test1)
lda1 = LinearDiscriminantAnalysis()
lda1.fit(X_kpca1, y_train1)
X_lda1 = lda1.transform(X_kpca1)
X_lda_test1 = lda1.transform(X_kpca_test1)


#kpca2 = KernelPCA(kernel="rbf",   n_components=6)
kpca2 = KernelPCA(kernel="rbf",   n_components=8)
#kpca2 = KernelPCA(kernel="rbf",   n_components=11)
kpca2.fit(X_train2)
X_kpca2 = kpca2.transform(X_train2)
X_kpca_test2 = kpca2.transform(X_test2)
lda2 = LinearDiscriminantAnalysis()
lda2.fit(X_kpca2, y_train2)
X_lda2 = lda2.transform(X_kpca2)
X_lda_test2 = lda2.transform(X_kpca_test2)

Dataset = SVSearch(X_train,X_test,y_train,y_test )
Dataset1 = SVSearch(X_train1,X_test1,y_train1,y_test1 )
Dataset2 = SVSearch(X_train2,X_test2,y_train2,y_test2 )


