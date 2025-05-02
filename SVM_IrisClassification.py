from sklearn.metrics import classification_report,accuracy_score   
from sklearn.datasets import load_iris   
from sklearn.preprocessing import StandardScaler   
import pandas as pd   
import numpy as np   
from sklearn.model_selection import train_test_split   
from sklearn.svm import SVC   
data = load_iris()   
x = data.data   
y = data.target   
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,  
random_state=42,stratify=y)  
scaler = StandardScaler()   
X_train = scaler.fit_transform(X_train)   
X_test = scaler.transform(X_test)   
svc_model = SVC(kernel='rbf'',C=0.1,gamma='scale',random_state=42)   
svc_model.fit(X_train,y_train)   
y_pred = svc_model.predict(X_test)   
print("classification report:\n",classification_report(y_test,y_pred))   
print("accuracy score:\n",accuracy_score(y_test,y_pred))  
prediction = svm_model.predict(new_data_scaled)     
print("\nPrediction for new data:", data.target_names[prediction[0]])