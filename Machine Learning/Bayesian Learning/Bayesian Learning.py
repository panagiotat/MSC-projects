from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score , recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
%matplotlib inline


#Download and import Dataset(remove headers and quotes inorder to make training harder)
newsgroups_train = fetch_20newsgroups(subset='train' , shuffle=True , random_state=27,remove=["headers" , 'quotes'] )
newsgroups_test = fetch_20newsgroups(subset='test',  shuffle=True , random_state=27,remove=["headers" , 'quotes'])


#Create Multinomial NB model and train it 
Multinomial_NB = Pipeline([ ('TfidfVectorizer', TfidfVectorizer()), ('MultinomialNB', MultinomialNB())])
Multinomial_NB.set_params(MultinomialNB__alpha= 0.1).fit(newsgroups_train.data, newsgroups_train.target)

#Predict the Test_set using the model we just created
y_predicted = Multinomial_NB.predict(newsgroups_test.data)
y_test = newsgroups_test.target


#Calculating the metrics
acc_score = accuracy_score(y_test, y_predicted)
prec_score = precision_score (y_test, y_predicted , average="macro")
rec_score = recall_score(y_test, y_predicted, average="macro")
f1_scor = f1_score(y_test, y_predicted, average="macro")

#Printing the metrics
print("Multinomial Naive Bayes classifier Accuracy is:", acc_score)
print ("Multinomial Naive Bayes classifier Precision is:",prec_score)
print("Multinomial Naive Bayes classifier Recall is:",rec_score)
print("Multinomial Naive Bayes classifier F1 Score is:", f1_scor)


#Plotting the Metrics
figure(num=None, figsize=(12, 10), dpi=250, facecolor='w', edgecolor='k')
matrix = confusion_matrix(newsgroups_test.target, y_predicted )
matrix = sns.heatmap(matrix,annot=True,cbar=False , cmap='Reds', fmt='g', xticklabels=newsgroups_test.target_names , yticklabels=newsgroups_test.target_names )

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Multinomial NB Confusion Matrix alpha= 0.1')