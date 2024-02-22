#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PROJECT ON SELENIUM


# In[9]:


pip install selenium


# In[13]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
#s = Service(r"C:\Users\swath\OneDrive\Desktop\chromedriver-win64")
webD = webdriver.Chrome(r"C:\Users\swath\Downloads\chromedriver.exe")
webD.get("https://www.amazon.in/Samsung-Galaxy-Smartphone-Titanium-Storage/dp/B0CQYGF1QY/ref=sr_1_1_sspa?dib=eyJ2IjoiMSJ9.Cvb5oQzTTBkyPpPhqlOvz7EOkchHQvkaDbo4APYoTqkyCkDMU0P0AGISBFc0w5QMLvahQc-e4W1EpHPQKrJKT0tnC9eKDwaIKs1yhz1h2QxFeCc7nsqdW_AXyDl6IfEsEaYNCiZMIZGUcZwg69dQmtK51d1ey6bB_1o9O03mfTo0kjyYaSiQT9W1I6d-HqJ0Ep1FfSunwYGWTtKbkF6l6WZR81yNXsJgStRW5EYuiV-p0TlKINP8URLKYjS2D9v1qduBlWWEKcIrrEqTnt5HAooc8vB14NbkFpj_bqvXESU.v7TOLXxZ5L8j24iavxyH9Lr4vZ19xm1r44KhvJ4i9K4&dib_tag=se&keywords=samsung%2Bfold&qid=1708522603&s=electronics&sr=1-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1")


# In[14]:


ele=webD.find_element(By.CLASS_NAME,'a-price-whole')


# In[17]:


ele


# In[20]:


l='/html/body/div[2]/div/div[5]/div[3]/div[4]/div[13]/div/div/div[4]/div[1]/span[2]/span[2]/span[2]'
s=webD.find_element(By.XPATH,l)


# In[21]:


x=s.text
print(x)


# In[24]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
#s = Service(r"C:\Users\swath\OneDrive\Desktop\chromedriver-win64\chromedriver-win64\chromedriver.exe")
webD = webdriver.Chrome("C:\\Users\\swath\\OneDrive\\Desktop\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe")
webD.get("https://www.flipkart.com/samsung-galaxy-z-fold5-cream-256-gb/p/itm5e9a1e2b0b9d9?pid=MOBGRS32ZCAVYQ7V&lid=LSTMOBGRS32ZCAVYQ7VEMYE2L&marketplace=FLIPKART&q=samsung%20fold&sattr[]=color&sattr[]=storage&st=color&otracker=search")


# In[25]:


z='/html/body/div[1]/div/div[3]/div[1]/div[2]/div[2]/div/div[4]/div[1]/div/div[1]'
p=webD.find_element(By.XPATH,z)


# In[26]:


y=(p.text)
print(y)


# In[27]:


if(x>y):
    print('price in amazon is more than price in flipkart')
else:
    print('price in flipkart is more than price in amazon')


# In[ ]:




