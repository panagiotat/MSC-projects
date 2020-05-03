from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score , recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier


#Load Dataset
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y= breast_cancer.target

#Splitting the dataset, creating 2 smaller. One for Train and one for Test.
#Train Set is 75% of the dataset and Test Set is 25%.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


#Creating a Decision Tree and train it
dec_tree = tree.DecisionTreeClassifier(criterion= "entropy" , max_depth= 10)
dec_tree = dec_tree.fit(X_train, y_train)

#Predict Test Set
y_predicted = dec_tree.predict(X_test)

#Calculating the metrics
acc_score = accuracy_score(y_test, y_predicted)
prec_score = precision_score (y_test, y_predicted, average='binary')
rec_score = recall_score(y_test, y_predicted, average='binary')
f1_scor = f1_score(y_test, y_predicted, average='binary')

#Printing the metrics
print("Decision Tree Accuracy is:", acc_score)
print ("Decision Tree Precision is:",prec_score)
print("Decision Tree Recall is:",rec_score)
print("Decision Tree F1 Score is:", f1_scor)

#Export Decision Tree inorder to make it pdf
#tree.export_graphviz(dec_tree , out_file = "entropy_10", filled=True, rounded=True, special_characters=True, feature_names  = breast_cancer.feature_names , class_names  = breast_cancer.target_names)


#Creating a Random Forest and train it
rand_forest = RandomForestClassifier(n_estimators=2, max_depth=5 , criterion="entropy")
rand_forest.fit(X_train,y_train)

#Predict Test Set
y_forest_predicted = rand_forest.predict(X_test)

#Calculating the metrics
acc_score_RF = accuracy_score(y_test, y_forest_predicted)
prec_score_RF = precision_score (y_test, y_forest_predicted, average='binary')
rec_score_RF = recall_score(y_test, y_forest_predicted, average='binary')
f1_scor_RF = f1_score(y_test, y_forest_predicted, average='binary')

#Printing the metrics
print("Random Forest Accuracy is:", acc_score_RF)
print ("Random Forest Precision is:",prec_score_RF)
print("Random Forest Recall is:",rec_score_RF)
print("Random Forest F1 Score is:", f1_scor_RF)