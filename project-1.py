#!/usr/bin/env python
# coding: utf-8

# In[2]:


from bs4 import BeautifulSoup
import requests
headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:32.0) Gecko/20100101 Firefox/32.0'}
url="https://www.vcsdata.com/hospitals-healthcare-in-india.html"

response=requests.get(url,headers=headers)
print('The status code is',response.status_code)


# In[4]:


print(response.text[:100])


# In[5]:


soup=BeautifulSoup(response.text,"html.parser")
print(soup.find('title').text)


# In[7]:


s=soup.find(class_='smllogo')
print(s)


# In[8]:


d=s.find('img')
print(d)


# In[9]:


d=soup.find_all('img')
print(d)


# In[10]:


for i in soup.find_all('img'):
    val=i.get('title')
print(i)


# In[11]:


x=soup.find_all(class_='innertitle')
print(x)


# In[14]:


d_1=[]
d_2=[]
d_3=[]

for i in x:
    h=i.find('h3')
    d1=h.text
    d_1.append(d1)
    print(d1)


# In[15]:


a=soup.find_all(class_='col-md-12 mb-2 mt-2')
print(a)


# In[16]:


for i in a:
    a1=i.find('span')
    d2=a1.text
    d_2.append(d2)
    print(d2)


# In[17]:


p=soup.find_all(class_='col-md-6')
print(p)


# In[18]:


for i in p:
    i.get('Industry')
    d3=i.text
    d_3.append(d3)
    print(d3)


# In[19]:


import pandas as pd
data=list(zip(d_1,d_2,d_3))


# In[20]:


k=pd.DataFrame(data,columns=['Company_name','Address','Industry_name'])
print(k)


# In[22]:


k.to_csv('project_data.csv')
print(k)


# In[23]:


k.to_excel('vcs_Data.xlsx',index=True)
k


# In[ ]:




