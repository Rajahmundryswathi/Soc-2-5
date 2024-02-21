#!/usr/bin/env python
# coding: utf-8

# In[8]:


from bs4 import BeautifulSoup
import requests

url = "https://crawler-test.com/"

response = requests.get(url)
print("The Status Code is ",response.status_code)
print(response.text)

#print("imported properly")


# In[9]:


from bs4 import BeautifulSoup
import requests

url = "https://crawler-test.com/"

response = requests.get(url)
print("The Status Code is ",response.status_code)
print(response.text[2])


# In[10]:


from bs4 import BeautifulSoup
import requests

url = "https://crawler-test.com/"

response = requests.get(url)
print("The Status Code is ",response.status_code)
print(response.text[:50])


# In[15]:


from bs4 import BeautifulSoup
import requests

url = "https://crawler-test.com/"

response = requests.get(url)
print("The Status Code is ",response.status_code)
print(response.text[:100])


# In[8]:


from bs4 import BeautifulSoup
import requests

url = "https://crawler-test.com/"

response = requests.get(url)
#print("The Status Code is ",response.status_code)
#print(response.text[:100])

# ! To get the title
#find(),find_all()
soup =  BeautifulSoup(response.text, 'html.parser')
print(soup.find('title').text)


# In[3]:


#! To get the heading 
heading = soup.find('h1')
print(heading.text)



# In[4]:


#! To find a tag
links = soup.find('a')
print(links)


# In[7]:


all_links = soup.find_all('a')
print(type(all_links))
print("-------------------------------")
for val in all_links:
    print(val)
print()


# In[13]:


# ! To find the element by Id
head=soup.find(id="header")
print(head)
print()


# In[14]:


a=head.find('a')
print(a)


# In[19]:


#To find the element based on class 
class_based = soup.find(class_="row side-collapsed")
print(class_based)


# In[25]:


class_based = soup.find(class_="panel-header")
print(class_based.text)



# In[30]:


h= soup.find_all('h3')
for i in h:
    print(i)


# In[45]:


headings =soup.find_all('div',{'class':'panel-header'})
for val in headings:
    h3 = val.find('h3')
    print(h3.text)


# In[46]:


s=soup.find_all(class_="panel")
print(s[1])


# In[47]:


print(h[1])


# In[48]:


#To get the link from description box

d=soup.find_all(class_='panel')
description=d[1]
for val in description.find_all('a'):
    print(val)


# In[54]:


for val in description.find_all('a'):
    print(val.get('href'))


# In[68]:


print(val['href'])


# In[71]:


for val in description.find_all('a'):
    print("https://crawler-test.com/"+str(val['href']))


# In[79]:


desc=soup.find_all(class_='panel')
l1=[]
description=desc[1]
for val in description.find_all('a'):
    str1="https://crawler-test.com"+str(val['href'])
    l1.append(str1)


# In[80]:


f1=open("links.txt",'w')
for val in range(len(l1)):
    f1.write(str(l1[val]))
f1.close()


# In[81]:


with open('links.txt','w') as f1:
    for val in range(len(l1)):
        f1.write(str(l1[val]+'\n'))


# In[ ]:




