from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score , recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from matplotlib.pyplot import figure

#Load dataset and normalise it using minmaxscaler

diabete = pd.read_csv("diabetes.csv") 
x = diabete.values[:,0:7]  
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
y = diabete.values[:,8]

#Splitting the dataset, creating 2 smaller. One for Train and one for Test.
#Train Set is 75% of the dataset and Test Set is 25%.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)



#Create Array to store the metrics
metrics = []
best_f1 = 0


#Repeat the k-NN process for 200 different k inorder to find the best k
for i in range (1,200):
    
    #Create and train the model
    neigh = KNeighborsClassifier(n_neighbors=i , p=1 , metric = "minkowski" ,weights ="distance")   
    neigh.fit(X_train, y_train) 
    y_predicted = neigh.predict(X_test)
    f1=f1_score(y_test, y_predicted, average='macro')
    
    #Finding the best k for F1_score
    if best_f1<f1 :
        best_f1=f1
        best_f1_k=i 
    #Store metrics into metrics Array
    metrics.append([accuracy_score(y_test, y_predicted) ,precision_score (y_test, y_predicted, average='macro') ,recall_score(y_test, y_predicted, average='macro') , f1_score(y_test, y_predicted, average='macro')])

    
#Printing the metrics
print("Best K for F1_score is:" , best_f1_k )
print("Accuracy is:" ,metrics[best_f1_k-1][0])
print("Precision score is:" ,metrics[best_f1_k-1][1])
print("Recall score is:" ,metrics[best_f1_k-1][2])
print("F1 score is:" ,metrics[best_f1_k-1][3])


#Normalise the metrics using minmaxscaler
scaler.fit(metrics)
metrics = scaler.transform(metrics)


#Plotting the Metrics
t = np.arange(1, 200, 1)
fig, axs = plt.subplots(4 ,1, constrained_layout=True)
fig.set_figheight(15)
fig.set_figwidth(15)

for ax in axs.flat:
    ax.set(xlabel='k-Nearest Neighbors')

axs[0].set_title('Accuracy' , color= "blue")
axs[0].plot(t, metrics[:,0] , color = "blue" )

axs[1].set_title('Precision', color= "orange")
axs[1].plot(t, metrics[:,1] , color = "orange")

axs[2].set_title('Recall', color= "green")
axs[2].plot(t, metrics[:,2] , color = "green")

axs[3].set_title('F1 Score', color= "red")
axs[3].plot(t, metrics[:,3] , color = "red")

plt.show()
figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
plt1.plot(t, metrics)
plt1.show()