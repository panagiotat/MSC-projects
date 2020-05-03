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
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.manifold import Isomap
from sklearn.metrics.cluster import homogeneity_score


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
def diagramm(method,clusters,train_set,y_set,name):
    res=[]
    clust = []
    for i in range(2,clusters):
        model = method(n_clusters=i, random_state=0).fit(train_set)
        a = homogeneity_score(y_set,model.labels_)
        res.append(a)
        clust.append(int(i))
        
        
    figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot( clust ,res, color="blue")
    plt.ylabel('Homogeneity Score')
    plt.xlabel('Clusters')
    plt.title(name)
    plt.show()



#Load the three datasets and shuffle them
train_set = pd.read_csv("2019.csv")
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

a=200
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
X_train, X_test, y_train, y_test = train_test_split(x_train , y, test_size=0.1)
X_train1, X_test1, y_train1, y_test1 = train_test_split(x_train1 , y1, test_size=0.1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(x_train2 , y2, test_size=0.1)

#Setting up the pre_processing models 
scaler = MinMaxScaler()
enc = OneHotEncoder( sparse="True"  )

#Preprocess the data
X_train , X_test = data_preprocessing1(dataset=X_train , test_dataset= X_test,  pairs= [ [["GDP per capita" , "Social support", "Healthy life expectancy" , "Freedom to make life choices", "Generosity", "Perceptions of corruption"],[scaler]] , [[],[enc]]])
X_train1 , X_test1 = data_preprocessing1(dataset=X_train1 , test_dataset= X_test1, pairs= [   [["carat", "depth" , "table","price", "x", "y","z"],[scaler]], [["color","clarity"],[enc]]])
X_train2 , X_test2 = data_preprocessing1(dataset=X_train2 , test_dataset= X_test2, pairs= [ [["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],[scaler]],  [[],[enc]]])


#Using Isomap to reduce dimension of the data inorder to plot them
embedding = Isomap(n_components=2)
X_transformed = embedding.fit_transform(X_train)
embedding1 = Isomap(n_components=2)
X_transformed1 = embedding1.fit_transform(X_train1)
embedding2 = Isomap(n_components=2)
X_transformed2 = embedding.fit_transform(X_train2)


#Plotting Data into 2 dimension scatter plot. Ground Truth, using KMeans and using Spectral Clustering
#Dataset1
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train)
spectral = SpectralClustering(n_clusters=3,random_state=0).fit(X_train)
fig = plt.figure(figsize=(8,8))
fig.suptitle("Dataset1 Ground Truth")
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=y_train)
fig = plt.figure(figsize=(8,8))
fig.suptitle("Dataset1 KMeans")
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=kmeans.labels_)
fig = plt.figure(figsize=(8,8))
fig.suptitle("Dataset1 Spectral Clustering")
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=spectral.labels_)
diagramm(KMeans,40,X_train,y_train,"KMeans")
diagramm(SpectralClustering,40,X_train,y_train,"Spectral Clustering")

#Dataset2
kmeans = KMeans(n_clusters=6, random_state=0).fit(X_train2)
spectral = SpectralClustering(n_clusters=6,random_state=0).fit(X_train2)
fig = plt.figure(figsize=(8,8))
fig.suptitle("Dataset2 Ground Truth")
plt.scatter(X_transformed2[:,0], X_transformed2[:,1], c=y_train2)
fig = plt.figure(figsize=(8,8))
fig.suptitle("Dataset2 KMeans")
plt.scatter(X_transformed2[:,0], X_transformed2[:,1], c=kmeans.labels_)
fig = plt.figure(figsize=(8,8))
fig.suptitle("Dataset2 Spectral Clustering")
plt.scatter(X_transformed2[:,0], X_transformed2[:,1], c=spectral.labels_)
diagramm(KMeans,40,X_train2,y_train2, "KMeans")
diagramm(SpectralClustering,40,X_train2,y_train2, "Spectral Clustering")

#Dataset3
kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train1)
spectral = SpectralClustering(n_clusters=5,random_state=0).fit(X_train1)
fig = plt.figure(figsize=(8,8))
fig.suptitle("Dataset3 Ground Truth")
plt.scatter(X_transformed1[:,0], X_transformed1[:,1], c=y_train1)
fig = plt.figure(figsize=(8,8))
fig.suptitle("Dataset3 KMeans")
plt.scatter(X_transformed1[:,0], X_transformed1[:,1], c=kmeans.labels_)
fig = plt.figure(figsize=(8,8))
fig.suptitle("Dataset3 Spectral Clustering")
plt.scatter(X_transformed1[:,0], X_transformed1[:,1], c=spectral.labels_)
diagramm(KMeans,40,X_train1,y_train1,"KMeans")
diagramm(SpectralClustering,40,X_train1,y_train1,"Spectral Clustering")






