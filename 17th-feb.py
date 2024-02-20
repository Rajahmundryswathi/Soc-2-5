#!/usr/bin/env python
# coding: utf-8

# In[135]:


# K means clustering


# In[274]:


#k means clustering is an umsupervised learning algorithm that will attempt 
#to group similar clusters together from your data 

#It is mainly used in 

#clustering similar documents
#clustering customers based on similar features


# In[275]:


from sklearn.datasets import make_blobs 
#The make_blobs functions is used to generates synthetic datasets 
#for clustering and classification tasks.
#This function will create clusters of data points with
#Gaussoam distribution 



# In[276]:


#creating random dataset
data =make_blobs(n_samples=200,n_features=2, centers =4,cluster_std=1.6,random_state=101)
#n_sample =Total number of points equally divided among clusters 
#n_features =It indicated the number of features(colulmns)
#centers =It determines no.of clusters to be generated 
#cluster_std =It sets the standard deviation of the clusters ,High value makes the clusters to be spread out


# In[277]:


data


# In[278]:


import matplotlib.pyplot as plt 
x,y = data
plt.scatter(x[:,0] , x[:,1] ,c=y)
plt.xlabel("Features 1")
plt.ylabel("Features 2")
plt.title("Scatters plot for K means")


# In[279]:


data[0].shape


# In[280]:


import matplotlib.pyplot as plt 
x,y = data
plt.scatter(x[:,0] , x[:,1] ,c=y ,cmap='rainbow' ,edgecolor='black' ,s=50)
plt.xlabel("Features 1")
plt.ylabel("Features 2")
plt.title("Scatters plot for K means")


# In[281]:


from sklearn.cluster import KMeans
kmeans =KMeans(n_clusters=4)
kmeans.fit(data[0])


# In[282]:


kmeans.cluster_centers_


# In[283]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
ax1.set_title('Kmeans') #predicted data
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_, cmap='rainbow')

ax2.set_title('Original')#Original data
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')


# In[356]:


#project:4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:\\Users\\swath\\Downloads\\College_Data")
df.head()


# In[330]:


df.info()


# In[331]:


# Total number of null or nan fields

df.isna().sum()


# In[332]:


df.isnull().sum()


# In[333]:


df.duplicated()


# In[334]:


if not df[df.duplicated()].empty:
    print(df[df.duplicated()])
else:
    print('no duplicate data')


# In[335]:


df['Private'].value_counts()


# In[336]:


df.index


# In[337]:


plt.figure(figsize=(3,3))
sns.barplot(x=df['Private'],y=df.index)


# In[338]:


plt.figure(figsize=(3,3))
sns.barplot(x=df.index,y=df['Private'])


# In[339]:


plt.figure(figsize=(3,3))
sns.barplot(x=df['Private'],y=df.index)
plt.xlabel("Private")
plt.ylabel("count")

plt.savefig("comaprison.png")


# In[340]:


sns.barplot(x=df['Private'],y=df['Grad.Rate'])


# In[341]:


sns.boxplot(x=df['Private'],y=df['Grad.Rate'])


# In[342]:


df[df['Grad.Rate']>100]


# In[343]:


df[df['Grad.Rate']>100]['Grad.Rate']


# In[344]:


df.loc[95,'Grade.Rate']=100


# In[345]:


h=df[df['Grad.Rate']>100]
h


# In[346]:


d1={"Grade_Rate":{"collage1":118 ,"collage2":100 } ,"a":200,"b":300}
d1


# In[347]:


#change value a to 400
d1["Grade_Rate"]["collage1"]=100
d1


# In[348]:


d1['a']=400
d1


# In[357]:


# from sklearn.cluster import KMeans
# kmeans=KMeans(n_clusters=2)
features =df.iloc[:,2:]
features


# In[358]:


#convert all columns into data type to string , to apply standardScaler()
features.columns = features.columns.astype(str)


# In[359]:


from sklearn.preprocessing import StandardScaler
#StandarScaler --->is a preprocessing class that is used to standardise
#or normalize the features of dataset.It has a mean of 0 and std of 1


# In[360]:


scaler=StandardScaler()
scaled_features=scaler.fit_transform(features)
scaled_features.shape


# In[361]:


from sklearn.cluster import KMeans
km=KMeans(n_clusters=2)
df['Cluster']=km.fit_predict(scaled_features)


# In[362]:


df['Cluster']


# In[364]:


df


# In[368]:


from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(df['Cluster'], km.labels_))


# In[370]:


print(accuracy_score(km.labels_,df['Cluster']))


# In[375]:


plt.scatter(km.labels_,features['P.Undergrad'])


# In[376]:


plt.bar(km.labels_,features['P.Undergrad'])


# In[377]:


plt.bar(df['Cluster'],features['P.Undergrad'])


# In[378]:


km.labels_


# In[ ]:





# In[ ]:





# In[1]:


# Diff b/w KNN and K means clustering

# 1)KNN is used for classification and reggression 
#K means is used for clustering problem

#2)KNN is supervised algorithm
# K means is unsupervised algorithm 

#3)To training KNN , we need a dataset with all the data
#points having class labels

#4)We use KNN to predict the class label or new points 
#We use K means to find the patterns in a given dataset by grouping datapoints into clusters


# In[4]:


import numpy as np
import pandas as pd

df = pd.read_csv("C:\\Users\\swath\\Downloads\\Classified Data")
df.head()


# In[6]:


df = pd.read_csv("C:\\Users\\swath\\Downloads\\Classified Data",index_col=0)
df


# In[7]:


#Project 5:
df = pd.read_csv("C:\\Users\\swath\\Downloads\\Classified Data",index_col='EQW')
df


# In[8]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(df.drop('TARGET CLASS',axis=1))


# In[14]:


Scaled_features = scalar.transform(df.drop("TARGET CLASS",axis=1))
Scaled_features


# In[17]:


df_feat = pd.DataFrame(Scaled_features)
df_feat


# In[21]:


#Example of standard scalar
data = np.array([[0,0],[0,1],[1,0],[1,1]])
data


# In[25]:


scl=StandardScaler()
scl


# In[26]:


scl_data =sc1.fit_transform(data)


# In[28]:


data


# In[29]:


scl_data


# In[31]:


scl_data.mean()


# In[32]:


scl_data.std()


# In[35]:


df.head() #original data


# In[37]:


df_feat.head() #scaled data


# In[44]:


#to name the column 
df_feat = pd.DataFrame(Scaled_features, columns=df.columns[:-1])
df_feat


# In[45]:


df_feat.isna().sum()


# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


x =df_feat
y =df['TARGET CLASS']


# In[48]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[49]:


x.shape


# In[50]:


x_train.shape


# In[51]:


x_test.shape


# In[60]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)


# In[61]:


KNN


# In[62]:


#To train model
KNN.fit(x_train,y_train)


# In[64]:


pred=KNN.predict(x_test)
pred


# In[65]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(pred,y_test)
acc


# In[76]:


# To find the error rate
error_rate= []
for val in range(1,30):
    knn=KNeighborsClassifier(n_neighbors=val)
    knn.fit(x_train,y_train)
    pred_i= knn.predict(x_test)
    error_rate.append(np.mean(pred_i!=y_test))
    


# In[77]:


error_rate


# In[79]:


import matplotlib.pyplot as plt
plt.figure(figsize=(6,3))
plt.plot(range(1,30), error_rate)
plt.title("Error rate VS Value")
plt.xlabel("K")
plt.ylabel("Error Rate")


# In[86]:


#Project : 6
import pandas as pd
df = pd.read_csv("C:\\Users\\swath\\Downloads\\cancerKNNAlgorithmDataset.csv")
df.head()


# In[ ]:




