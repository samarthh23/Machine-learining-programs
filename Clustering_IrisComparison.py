
import pandas as pd   
import numpy as np   
from sklearn.metrics import silhouette_score   
from sklearn.datasets import load_iris   
import matplotlib.pyplot as plt   
from sklearn.cluster import KMeans   
from sklearn.mixture import GaussianMixture   
iris = load_iris()   
x = pd.DataFrame(iris.data,columns = iris.feature_names)   
y = iris.target   
plt.figure(figsize=(10,6))   
plt.subplot(1,2,1)   
plt.scatter(x[’sepal length (cm)’],x[’sepal width (cm)’],c=y,s=40)   
plt.title("actual sepal")   
plt.subplot(1,2,2)   
plt.scatter(x[’petal length (cm)’],x[’petal width (cm)’],c=y,s=40)   
plt.title("actual petal")   
kmeans = KMeans(n_clusters=3)   
kmeans.fit(x)  
kmeans_clusters = kmeans.labels_   
gmm = GaussianMixture(n_components=3)   
gmm.fit(x)   
gmm_clusters = gmm.predict(x)   
kmeans_silhouette = silhouette_score(x,kmeans_clusters)   
gmm_silhouette = silhouette_score(x,gmm_clusters)   
print("kmeans_silhouette :",kmeans_silhouette)   
print("gmm_silhouette :",gmm_silhouette)   
plt.figure(figsize=(10,6))   
plt.subplot(1,3,1)   
plt.scatter(x[’sepal length (cm)’],x[’petal width (cm)’],c=y,s=40)   
plt.title("actual classes")   
plt.subplot(1,3,2)   
plt.scatter(x[’sepal length (cm)’],x[’petal width (cm)’],c=kmeans_clusters,s=40)   
plt.title("kmeans")   
plt.subplot(1,3,3)   
plt.scatter(x[’sepal length (cm)’],x[’petal width (cm)’],c=gmm_clusters,s=40)   
plt.title("gmm")   
plt.show()   
if kmeans_silhouette>gmm_silhouette:   
print("kmeans")   
else:   
print("gmm") 