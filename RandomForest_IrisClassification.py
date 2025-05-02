import numpy as np   
from sklearn.metrics import classification_report,accuracy_score   
from sklearn.datasets import load_iris   
from sklearn.model_selection import train_test_split   
import matplotlib.pyplot as plt   
from sklearn.ensemble import RandomForestClassifier   
iris = load_iris()   
x = iris.data   
y = iris.target   
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)   
rf_classifier = RandomForestClassifier(n_estimators=100,random_state=42)   
rf_classifier.fit(X_train,y_train)   
y_pred = rf_classifier.predict(X_test)   
print(classification_report(y_test,y_pred))   
print(accuracy_score(y_test,y_pred))   
feature_importances = rf_classifier.feature_importances_   
features = iris.feature_names   
plt.figure(figsize=(10,6))   
plt.barh(features,feature_importances, color = ’skyblue’)   
plt.xlabel("feature importances")   
plt.show() 