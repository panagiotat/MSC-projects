from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score , recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3


#Load dataset and seperate input and output
tweets = pd.read_csv("hate_tweets.csv") 
x = tweets.values[:,6]
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x)
y= tweets.values[:,2:5]
y = np.argmax(y, axis=1)
y = np.eye(3)[y]

#Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

#Creating a Decision Tree and train it
dec_tree = tree.DecisionTreeClassifier(criterion= "entropy" , max_depth= 20)
dec_tree = dec_tree.fit(X_train, y_train)

#Creating a Decision Tree and train it
dec_tree_gini = tree.DecisionTreeClassifier(criterion= "gini" , max_depth= 20)
dec_tree_gini = dec_tree_gini.fit(X_train, y_train)

#Predict Test Set
y_predicted = dec_tree.predict(X_test)
y_predicted_gini = dec_tree_gini.predict(X_test)

#Calculating the metrics
acc_score = accuracy_score(y_test, y_predicted)
prec_score_RF = precision_score (y_test, y_forest_predicted, average='micro')
rec_score_RF = recall_score(y_test, y_forest_predicted, average='micro')
f1_scor_RF = f1_score(y_test, y_forest_predicted, average='micro')

acc_score_gini = accuracy_score(y_test, y_predicted_gini)
prec_score_gini = precision_score (y_test, y_predicted_gini, average='micro')
rec_score_gini = recall_score(y_test, y_predicted_gini, average='micro')
f1_scor_gini = f1_score(y_test, y_predicted_gini, average='micro')


#Printing the metrics
print("Decision Tree with Depth 20 and Entropy as metric Accuracy is:", acc_score)
print ("Decision Tree with Depth 20 and Entropy as metric Precision is:",prec_score)
print("Decision Tree with Depth 20 and Entropy as metric Recall is:",rec_score)
print("Decision Tree with Depth 20 and Entropy as metric F1 Score is:", f1_scor)

print("Decision Tree with Depth 20 and Gini as metric Accuracy is:", acc_score_gini)
print ("Decision Tree with Depth 20 and Gini as metric Precision is:",prec_score_gini)
print("Decision Tree with Depth 20 and Gini as metric Recall is:",prec_score_gini)
print("Decision Tree with Depth 20 and Gini as metric F1 Score is:", prec_score_gini)


result_acc = []
prec_score_res = []
rec_score_res = []
f1_scor_res = []

result_acc_gini = []
prec_score_res_gini = []
rec_score_res_gini = []
f1_scor_res_gini = []

number_of_trees = []


for i in range(1,40,1):
    #Creating a Random Forest and train it
    rand_forest = RandomForestClassifier(n_estimators=i, max_depth=10 , criterion="entropy")
    rand_forest_gini = RandomForestClassifier(n_estimators=i, max_depth=10 , criterion="gini")
    rand_forest.fit(X_train,y_train)
    rand_forest_gini.fit(X_train,y_train)

    #Predict Test Set
    y_forest_predicted = rand_forest.predict(X_test)
    y_forest_predicted_gini = rand_forest_gini.predict(X_test)
    
    #Calculating the metrics
    acc_score_RF = accuracy_score(y_test, y_forest_predicted)
    prec_score_RF = precision_score (y_test, y_forest_predicted, average='micro')
    rec_score_RF = recall_score(y_test, y_forest_predicted, average='micro')
    f1_scor_RF = f1_score(y_test, y_forest_predicted, average='micro')
    
    acc_score_RF_gini = accuracy_score(y_test, y_forest_predicted_gini)
    prec_score_RF_gini = precision_score (y_test, y_forest_predicted_gini, average='micro')
    rec_score_RF_gini = recall_score(y_test, y_forest_predicted_gini, average='micro')
    f1_scor_RF_gini = f1_score(y_test, y_forest_predicted_gini, average='micro')
    
    
    #Store results
    result_acc.append(acc_score_RF)
    prec_score_res.append(prec_score_RF)
    rec_score_res.append(rec_score_RF)
    f1_scor_res.append(f1_scor_RF)
    
    result_acc_gini.append(acc_score_RF_gini)
    prec_score_res_gini.append(prec_score_RF_gini)
    rec_score_res_gini.append(rec_score_RF_gini)
    f1_scor_res_gini.append(f1_scor_RF_gini)
    
    number_of_trees.append(i)

print("Red is Entropy diagram and blue is Gini")
figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
plt.plot(number_of_trees, result_acc , color = "red")
plt.plot(number_of_trees, result_acc_gini , color = "blue")
plt.ylabel('Test Accuracy')
plt.xlabel('Trees')
plt.show()

print("Red is Entropy diagram and blue is Gini")
figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
plt1.plot(number_of_trees, prec_score_res , color = "red")
plt1.plot(number_of_trees, prec_score_res_gini , color = "blue")
plt1.ylabel('Test Precision Accuracy')
plt1.xlabel('Trees')
plt1.show()

print("Red is Entropy diagram and blue is Gini")
figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
plt2.plot(number_of_trees, rec_score_res , color = "red")
plt2.plot(number_of_trees, rec_score_res_gini , color = "blue")
plt2.ylabel('Test Recall Accuracy')
plt2.xlabel('Trees')
plt2.show()

print("Red is Entropy diagram and blue is Gini")
figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
plt3.plot(number_of_trees, f1_scor_res , color = "red")
plt3.plot(number_of_trees, f1_scor_res_gini , color = "blue")
plt3.ylabel('Test F1 Accuracy')
plt3.xlabel('Trees')
plt3.show()
