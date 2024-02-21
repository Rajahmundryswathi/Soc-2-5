#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df1=pd.read_csv("C:\\Users\\swath\\Downloads\\titanic_train.csv")
df1


# In[2]:


import matplotlib.pyplot as plt
df1[df1['Pclass']==2]['Pclass'].sum()


# In[3]:


class1 = df1[df1['Pclass']==1]['Age'].dropna()


# In[4]:


class2=df1[df1['Pclass']==2]['Age'].dropna()


# In[5]:


class3=df1[df1['Pclass']==3]['Age'].dropna()


# In[6]:


l1=[class1,class2,class3]
plt.boxplot(l1,labels=['class1','class2','class3'])


# In[7]:


#matplotlib inline 
df1.rename(columns={'Sex':'Gender'},inplace=True)
df1


# In[8]:


#to change the gender field male as 0 to female as 1
df1['Gender']=df1['Gender'].map({'male':0,'female':1})
df1


# In[9]:


df1['Gender']=df1['Gender'].replace({'male':0,'female':1})
df1


# In[10]:


df1.head(5)


# In[11]:


((df1['Age']<25) & (df1['Gender']==1)).sum()


# In[12]:


import seaborn as sns
tip=sns.load_dataset('tips')
tip


# In[13]:


#-->displot - it will be take only one column
plt.figure(figsize=(2,3))
sns.distplot(tip['total_bill'],bins=100,kde=False)


# In[14]:


plt.figure(figsize=(2,3))
sns.distplot(tip['total_bill'])


# In[15]:


plt.figure(figsize=(2,3))
sns.distplot(tip['total_bill'],bins=100,kde=False,hist=True,color='red')


# In[16]:


plt.figure(figsize=(2,3))
sns.distplot(tip['total_bill'],bins=100,kde=True,color='red')


# In[17]:


#jointplot
sns.displot(tip['total_bill'],bins=100,kde=True,color='blue')


# In[18]:


sns.jointplot(x='total_bill',y='tip',data=tip,kind='hex')


# In[19]:


sns.jointplot(x='total_bill',y='tip',data=tip,kind='reg')


# In[20]:


sns.jointplot(x='total_bill',y='tip',data=tip,kind='kde')


# In[21]:


#pairplot
sns.pairplot(tip,hue='sex')


# In[22]:


sns.pairplot(tip,hue='sex',palette='coolwarm')


# In[23]:


#logistic regression
#logistic regression is a statistical method used to model the relationship b/w a binary dependent 
#and one at more independent variables

#In logistic regression the dependent variablr is binary ,meaning it can only take on 2 values ,labelled as 0 or 1
#the independent variable can be either continuous or catagorical 


# In[24]:


import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix


# In[25]:


pip install scikit-learn


# In[26]:


import numpy as np
y_pred=np.array([0.3,0.6,0.8,0.2,0.4,0.9,0.1,0.7,0.5,0.6])
y_true=np.array([0,1,1,0,0,1,0,1,0,1])


# In[27]:


#Accuracy
#Accuracy measures the percentage of the correctly classified 
#instances out of all the instances 

accuracy = accuracy_score(y_true,np.round(y_pred))
accuracy


# In[28]:


#precision 
#precision measured the propotion of true positive prediction 
#out of all the positive prediction
#precision=true positive/all positive

precision=precision_score(y_true,np.round(y_pred))
precision


# In[29]:


#recall 
#recall measures the propotion of true positive prediction out
#of all actual position cases
#recall=true positive/actual cases

recall=recall_score(y_true,np.round(y_pred))
recall



# In[30]:


#f1 score
#It is the man of precision and recall
f1_score=f1_score(y_true,np.round(y_pred))
f1_score


# In[31]:


#confusion matrix
#It is a table that gives the performance of a classification model 
#It shows true positive , true negative , false positive ,false negative
matrix=confusion_matrix(y_true,np.round(y_pred))
matrix


# In[32]:


#eq:1
#logistic regression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
iris=load_iris()
iris


# In[33]:


iris_df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df


# In[34]:


iris_df['target']=iris.target_names[iris.target]
iris_df


# In[35]:


x=iris.data
x


# In[36]:


y=iris.target
y


# In[37]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)


# In[38]:


x_train.shape


# In[39]:


x_test.shape


# In[40]:


#to train the algorithm
clf=LogisticRegression()
clf


# In[41]:


clf.fit(x_train,y_train)


# In[42]:


y_pred=clf.predict(x_test)
y_pred


# In[43]:


accuracy==accuracy_score(y_test,y_pred)
accuracy


# In[44]:


#project 1
data=pd.read_csv("C:\\Users\\swath\\Downloads\\bmi.csv")


# In[45]:


data


# In[46]:


data.head()


# In[47]:


data['Gender'].value_counts()


# In[48]:


(data['Gender']=="Male").sum()


# In[49]:


(data['Gender']=="Female").sum()


# In[50]:


import pandas as pd
data=pd.get_dummies(data,columns=['Gender'],dtype=int)


# In[51]:


data


# In[52]:


import seaborn as sns
sns.barplot(y="Height",x="Index",data=data)


# In[66]:


import numpy as np
import pandas as pd
a=pd.read_csv("C:\\Users\\swath\\Downloads\\bmi.csv")
a


# # x1=x.drop("Index",axis=1)

# In[67]:


x1=a.drop("Index",axis=1)
x1


# In[69]:


b=x['Index']


# In[70]:


b


# In[71]:


a.shape


# In[72]:


b.shape


# In[73]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.3,random_state=0)


# In[74]:


a_train.shape


# In[75]:


b_train.shape


# In[76]:


a_test.shape


# In[77]:


l=LogisticRegression()


# In[78]:


l


# In[79]:


#to train the dataset
l.fit(a_train,b_train)


# In[80]:


pred=l.predict(a_test)
pred


# In[85]:


a=pd.get_dummies(x,columns=['Gender'],dtype=int,drop_first=True)
a


# In[86]:


accuracy_score(b_test,pred)


# In[ ]:


#linear regression
#Data:
#X (week)                        Y(sales in thousand)
#-----------------------------------------------------
#1                                  1.2
#2                                  1.8
#3                                  2.5
#4                                  3.2
#5                                  3.8

#linear regression formula --> y=a0+a1*x

#a0-->(meanof(x*y)) - ((x)*(y))/meanof(x^2) - ((x)^2)
#a0 = mean(y) - a1*meanof(x)
       
    # x     y          x^2    x*y
    # 1     1.2         1      1.2
    # 2     1.8         4      3.6
    # 3     2.5         9      7.5
    # 4     3.2         16     12.8
    # 5     3.8         25     19

#---------------------------------------------------------
#sum:15     12.5       55      44.1
#avg:x=3    y=2.5    x^2=11   x*y=8.82


#a1=(8.8 - (3*2.5)/11-3^2)
#a0=2.52-(0.66*3) = (0.54)


#The sales of 3rd week 
#y=a0+(a1*3)
#y=0.54+(0.66*3)
#y=2.52


#The sales for  7r week
#y=a0+a1*x
#y=0.54+0.66*7
#y=5.16


# In[87]:


#linear regression
#practice
import pandas as pd
s=pd.read_csv("C:\\Users\\swath\\Downloads\\Linear_regr_Salary_dataset.csv")
s


# In[88]:


s.shape


# In[89]:


s.isnull().sum()


# In[90]:


s.isna().sum()


# In[95]:


x=s[['YearsExperience']]


# In[96]:


y=s[['Salary']]


# In[97]:


from sklearn.model_selection import train_test_split


# In[98]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)


# In[99]:


from sklearn.linear_model import LinearRegression


# In[100]:


model=LinearRegression()
model


# In[101]:


model.fit(x_train,y_train)


# In[102]:


y_pred=model.predict(x_test)


# In[103]:


y_pred


# In[104]:


y_test


# In[105]:


acc=accuracy_score(y_test,np.round(y_pred))
acc


# In[106]:


inputdata=[[17]]
prediction=model.predict(inputdata)
prediction


# In[107]:


from sklearn.metrics import mean_squared_error


# In[108]:


mse=mean_squared_error(y_test,y_pred)


# In[109]:


mse


# In[110]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.lmplot(x='YearsExperience',y="Salary",data=s)


# In[111]:


sns.lmplot(x='YearsExperience',y="Salary",data=s)
ax=plt.gca()
plt.gca().set_facecolor('black')


# In[112]:


sns.lmplot(x='YearsExperience',y="Salary",data=s)
sns.set_style("darkgrid")
ax=plt.gca()

ax.tick_params(axis='x',colors='white')
ax.tick_params(axis='y',colors='white')
plt.gca().set_facecolor('black')


# In[113]:


sns.lmplot(x='YearsExperience',y="Salary",data=s,
scatter_kws={"color":"yellow"},line_kws={'color':'red'})
sns.set_style("darkgrid")
ax=plt.gca()

#ax.tick_params(axis='x',colors='white')
#ax.tick_params(axis='y',colors='white')
plt.gca().set_facecolor('black')


# In[160]:


#project:3
df=pd.read_csv("C:\\Users\\swath\\Downloads\\LR_Student_Performance (1).csv")
df


# In[115]:


df.shape


# In[116]:


df.isnull().sum()


# In[117]:


df.isna().sum()


# In[118]:


#To check duplicate values
duplicate_rows=df.duplicated()
duplicate_rows.sum()


# In[119]:


print("before",df.shape)
df.drop_duplicates(inplace=True)
print("after",df.shape)


# In[120]:


#based on index value try to check the performance
response =df['Performance Index']
response  


# In[121]:


plt.plot(response.index,response)
plt.plot("index")
plt.plot("Performance Index")


# In[122]:


sns.jointplot(x=response.index,y='Performance Index',data=df,kind='hex')


# In[123]:


sns.violinplot(response)


# 

# In[124]:


p=df['Performance Index'].max()
p


# In[125]:


q=df['Performance Index'].min()
q


# In[126]:


x=df['Hours Studied'].value_counts()
x


# In[127]:


x=df['Performance Index'].value_counts()
x


# In[128]:


x=df['Performance Index'].value_counts().min()
x


# In[129]:


x=df['Performance Index'].value_counts().max()
x


# In[130]:


#to get the all the unique values
df['Hours Studied'].unique()


# In[131]:


#To know how many students studies in each hours
df1=df[df['Performance Index']==100]
df1


# In[132]:


#To know the how many students in each hours
import seaborn as sns
x=df['Hours Studied']
sns.histplot(x,color='green')


# In[133]:


import seaborn as sns
x=df['Hours Studied']
sns.histplot(x,color='green',kde=True)


# In[134]:


import seaborn as sns
x=df['Sleep Hours']
sns.histplot(x,color='green')


# In[135]:


sns.boxplot(x=df['Hours Studied'],y=df['Performance Index'])


# In[136]:


x=df['Extracurricular Activities']
sns.histplot(x,color='green')


# In[137]:


x=df['Extracurricular Activities']
sns.histplot(x,color='green',kde=True)


# In[138]:


#sns.countplot(df['Extracurricular Activities'],colors='blue')


# In[139]:


df.columns


# In[163]:


x=df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities',
       'Sleep Hours', 'Sample Question Papers Practiced']]
x


# In[164]:


y=df['Performance Index']
y


# In[165]:


df.head()


# In[162]:


df['Extracurricular Activities']=df['Extracurricular Activities'].apply(lambda x:1 if x=='Yes' else 0)
df.head()


# In[166]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[167]:


x.shape


# In[168]:


y.shape


# In[169]:


model


# In[170]:


model.fit(x_train,y_train)


# In[171]:


y_pred=model.predict(x_test)
y_pred


# In[ ]:





# In[172]:


#Instead of linear regression 
#redge
from sklearn.linear_model import Ridge


# In[173]:


clf=Ridge()


# In[174]:


clf.fit(x_train,y_train)


# In[182]:


y_pred=clf.predict(x_test)
y_pred


# In[177]:


clf.score(x_test,y_test)


# In[ ]:





# In[184]:


y_pred=model.predict(x_test)
y_pred


# In[179]:


acc=accuracy_score(y_test,np.round(y_pred))
acc


# In[187]:


student=[[8,85,1,6,6,7]]
prediction=model.predict(x)
prediction


# In[ ]:





# In[ ]:




