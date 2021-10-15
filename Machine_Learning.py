#KNN - Supervised Learning
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


df = pd.read_csv("Stars.csv")
#Set A_M (Manitude) and Spectral_Class to numerical values
for col in df.columns:
  df[col]=df[col].astype('category').cat.codes


X=df.loc[:,df.columns != 'Type']
y=df.loc[:,['Type']]

#Create Train Val and Test (60/20/20)
X_T, X_test, y_T, y_test = train_test_split(X, y, test_size=0.20)
X_train, X_val, y_train, y_val = train_test_split(X_T, y_T, test_size=0.25)

#Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


best_k=0
best_acc=0
accs = []
# Calculating error for K values between 1 and 20
for j in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=j)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_val)
    acc = accuracy_score(pred_i, y_val)
    accs.append(acc)
    #Automatically select best k
    if(acc>=best_acc):
        best_acc=acc
        best_k=j
#Plot accuracy for each K
print("Best K is ",best_k)
plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), accs, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
plt.title('Accuracy Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.show()

#Test
classifier = KNeighborsClassifier(n_neighbors=best_k)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Kmeans

#Hierarchical Clustering 