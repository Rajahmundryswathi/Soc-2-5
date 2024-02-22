#!/usr/bin/env python
# coding: utf-8

# In[14]:


pip install selenium


# In[15]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
#s = Service(r"C:\Users\swath\Downloads\chromedriver.exe")
driver = webdriver.Chrome(r"C:\Users\swath\Downloads\chromedriver.exe")
driver.get("https://books.toscrape.com/")


# In[16]:


import pandas as pd
raw_mail_data=pd.read_csv("C:\\Users\\swath\\Downloads\\mail_data.csv")
raw_mail_data.head()


# In[17]:


#To check the mail in spam 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score


# In[18]:


#To check and Nan field in column
raw_mail_data.isna().sum()


# In[19]:


#To convert catagory field to numerical field 'spam'= 0 
raw_mail_data.loc[raw_mail_data['Category']=='spam','Category']=0
raw_mail_data['Category']


# In[20]:


#To convert category field to numerical field 'ham'= 1
raw_mail_data.loc[raw_mail_data['Category']=='ham','Category']=1
raw_mail_data['Category']


# In[21]:


Y=raw_mail_data['Category']
X=raw_mail_data['Message']


# In[22]:


X


# In[23]:


#Convert to train and test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)


# In[24]:


x_train.shape


# In[25]:


x_test.shape


# In[26]:


y_train.shape


# In[27]:


y_test.shape


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[32]:


feature_extraction=TfidfVectorizer()
x_train_feature=feature_extraction.fit_transform(x_train)
print(x_train_feature)


# In[33]:


x_test_feature=feature_extraction.transform(x_test)
print(x_test_feature)


# In[34]:


y_train=y_train.astype('int')
y_test=y_test.astype('int')


# In[35]:


y_train.dtype


# In[36]:


#Create instance for Logostic regression
model=LogisticRegression()


# In[37]:


model


# In[38]:


model.fit(x_train_feature,y_train)


# In[39]:


#Process of training the model
prediction = model.predict(x_test_feature)


# In[40]:


prediction


# In[41]:


acc=accuracy_score(prediction,y_test)


# In[42]:


print("The accuracy of the above model is =",acc)


# In[43]:


#-->real time verification
#input mail=[""]
raw_mail_data['Message'][467]


# In[44]:


raw_mail_data['Category'][467]


# In[45]:


input_mail=["They dont put that stuff on the roads to keep it from getting slippery over there"]


# In[46]:


input_data_feature=feature_extraction.transform(input_mail)
print(input_data_feature)


# In[47]:


Prediction = model.predict(input_data_feature)
if prediction[0]==1:
    print("Its Ham mail")
else:
    print("Its Spam mail")


# In[48]:


#TO retrieve the elements from webpage

#To find element by name attribute
#element = webD.find_element_by_name("element name")

#To find element by link
#element = webD.find_element_by_link('link')

#To find element by tagname
#element = webD.find_element_by_tag_name('link')

#To find element by xpath
# element=webD.find_element_by_tag_xpath('xpath')

# #-->find element by classname
# element=webD.find_element_by_class_name('classname')

# #-->find element by id
# element=webD.find_element_by_id('id')


# In[49]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
#s = Service(r"C:\Users\swath\Downloads\chromedriver.exe")
webD = webdriver.Chrome(r"C:\Users\swath\Downloads\chromedriver.exe")
webD.get("https://books.toscrape.com/")


# In[50]:


ele = webD.find_element(By.CLASS_NAME, "col-sm-8.h1")
print(ele)


# In[51]:


a=webD.find_element(By.TAG_NAME,'a')


# In[52]:


print(a.text)


# In[53]:


#!To get all the books category 
class_name = webD.find_element(By.CLASS_NAME,"side_categories")
list1 = class_name.find_elements(By.TAG_NAME,'li')
for val in list1:
    print(val.text)


# In[ ]:




