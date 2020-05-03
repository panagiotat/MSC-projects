import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler , OneHotEncoder
from sklearn.metrics import explained_variance_score , mean_squared_error, mean_absolute_error , r2_score
from sklearn.neighbors import KNeighborsRegressor


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



for i in range (1,31):
    
    neigh = KNeighborsRegressor(n_neighbors=i)
    neigh.fit(X_train, y_train) 


    y_predict_test = neigh.predict(X_test)
    t1 = mean_absolute_error(y_predict_test , y_test)
    print("With",i , "Neighbors the absolute error at test set is:",t1)

    
    
    
    


