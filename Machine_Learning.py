import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("Stars.csv")
#Set A_M (Manitude) and Spectral_Class to numerical values
for col in df.columns:
  df[col]=df[col].astype('category').cat.codes


X=df.loc[:,df.columns != 'Type']
y=df.loc[:,['Type']]

#---KNN---
#Create Train Val and Test (60/20/20)
X_T, X_test, y_T, y_test = train_test_split(X, y, test_size=0.20)
X_train, X_val, y_train, y_val = train_test_split(X_T, y_T, test_size=0.25)

#Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#Find best k
best_k=0
best_acc=0
accs = []
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




#---OnevsAll---
X_T, X_test, y_T, y_test = train_test_split(X, y, test_size=0.20)
X_train, X_val, y_train, y_val = train_test_split(X_T, y_T, test_size=0.25)
Multiclass_model = LogisticRegression(multi_class='ovr')
Multiclass_model.fit(X_train, y_train)

y_pred = Multiclass_model.predict(X_val)
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

y_pred = Multiclass_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



#---Naive Bayes---
X_T, X_test, y_T, y_test = train_test_split(X, y, test_size=0.20)
X_train, X_val, y_train, y_val = train_test_split(X_T, y_T, test_size=0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred  =  classifier.predict(X_val)
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

y_pred  =  classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



#---Kmeans---
#Find K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Kmeans
kmeans = KMeans(n_clusters=6)
kmeans.fit(X)

#Plot truth

#Plot Predicted
nb_groupe = 6
#Load Data
#Transform the data
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
df = pca.fit_transform(X)
#Initialize the class object
kmeans = KMeans(n_clusters= nb_groupe)
#predict the labels of clusters.
label = kmeans.fit_predict(df)
#Getting unique labels
u_labels = np.unique(label)
#plotting the results:
for i in u_labels:
    plt.scatter(df[label == i , 1] , df[label == i , 0] , label = i)
plt.title('Predicted')
plt.legend()
plt.show()
