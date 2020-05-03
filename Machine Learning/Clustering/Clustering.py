from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

breast_cancer = load_breast_cancer()
scaler = MinMaxScaler()
scaler.fit(breast_cancer.data)
norm_data = scaler.transform(breast_cancer.data)



kmeans = KMeans(n_clusters=2, random_state=0).fit(norm_data)
cluster_labels = kmeans.predict(norm_data)
silhouette_avg = silhouette_score(norm_data, cluster_labels)
results = []
results.append( [2,silhouette_avg])
i=3
while silhouette_avg>0.1 :
    
    kmeans = KMeans(n_clusters=i, random_state=0).fit(norm_data)
    cluster_labels = kmeans.predict(norm_data)
    silhouette_avg = silhouette_score(norm_data, cluster_labels)
    results.append([i,silhouette_avg])
    i=i+1
    
    
results = np.array(results)


plt.plot(results[:,0], results[:,1])
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.show()


results = []
for i in range(2,100) :
    
    dbscn = DBSCAN( min_samples=i).fit(norm_data)
    cluster_labels = dbscn.labels_
    silhouette_avg = silhouette_score(norm_data, cluster_labels)
    results.append([i,silhouette_avg])
    i=i+1
    

results = np.array(results)

plt.plot(results[:,0], results[:,1])
plt.xlabel('Neighborhood ')
plt.ylabel('Silhouette Score')
plt.show()
