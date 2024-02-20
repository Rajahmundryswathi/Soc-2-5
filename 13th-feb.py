#!/usr/bin/env python
# coding: utf-8

# In[1]:


#how to create range of elements as array
#how to convert a 1d array to multidimensional array
#how create array of identity matrix
#how to perfom slicing in multidimensional array
#how to find the mean,median,sum,variance,standard deviation of an array


# In[2]:


#how to create a range of an element 
import numpy as np
n=np.arange(4,10)
n


# In[7]:


n=np.arange(4,10,2)
n


# In[64]:


l1=[1,2,3,4,5,6,7,8,9]
ar=np.array(l1)
ar


# In[65]:


#how to create a 1d array to multidimensional array
n=ar.reshape(3,3)
n


# In[14]:


#-->eye() is used to define both rows and columns
np.eye(3,4)


# In[57]:


#creating a identity matrix
#-->identity() is used to define same no.of rows and columns
import numpy as np
ar1=np.identity(4,dtype=int)
ar1


# In[66]:


#slicing in multidimensional array
n[1,1:4]


# In[67]:


#mean of  an array
ar2=np.mean(ar)
ar2


# In[68]:


#median of an array
ar2=np.median(ar)
ar2


# In[69]:


#sum of an array
ar2=np.sum(ar)
ar2


# In[70]:


#variance of an array
ar2=np.var(ar)
ar2


# In[15]:


#standard deviation of an array
ar3=np.std(ar)
ar3


# In[20]:


import numpy as np
n=np.arange(25).reshape(5,5)
n


# In[23]:


#[row_start:row_end,column_start:column_end,step]
n[0:3,1:4]


# In[24]:


n[1:3]


# In[28]:


n[0:3,1:4]


# In[36]:


n[-3:-1]


# In[41]:


n[-3:-1,-3:-1]


# In[ ]:


# 1.print 27,28,37,38-->both positive and negative
# 2.print 34,35,44,45
# 3.print 18,19
# 4.print 31,31,40,40


# In[44]:


n1=np.arange(50).reshape(5,10)
n1


# In[47]:


n2=np.sum(n1,axis=0)
n2


# In[ ]:


mean-->average
median-->centre value
mode-->max repeated values
variance-->median/total no.of values
std-->sqrt of variance


# In[ ]:





# In[48]:


#lenspace()
#--> it is used to print the evenly separated values
#(1,10,4,dtype=int)-->linspace(start,end,num of ele to be displayed)

np.linspace(1,5)


# In[49]:


np.linspace(1,10,5)


# In[52]:


np.linspace(1,10,5,retstep=True)


# In[71]:


#random module
#to generate random values b/w the range
#np.random.randint(start,end)
#np.random.randint(start,end,no.of val)
#np.random.randint(40)

np.random.randint(40,50)


# In[82]:


np.random.randint(40,50,2)


# In[83]:


#rand()-->
#to get the randomly generated values from 0 to 1,based on uniform distribution

np.random.rand(4)


# In[86]:


#randn()-->
#to get the randomly generated values from 0 to 1,based on normalized distribution

np.random.randn(5,3)


# In[92]:


n1=np.arange(0,50).reshape(5,10)
n1


# In[93]:


#find min val
n1.min()


# In[94]:


#max val
n1.max()


# In[101]:


#find the index of an ele
l1=[1,2,3,4,5,6]
l2=np.array(l1)
print(l1.index(2))


# In[96]:


#tolist convert array into list
print(n1.tolist())


# In[103]:


#argmin()-->to find the min val of index in an array
#argmin()-->to find the max val of index in an array
l1=[1,2,3,4,5,6]
l2=np.array(l1)
print(l2.argmin())
print(l2.argmax())


# In[104]:


np.sin(l2)
np.cos(l2)
np.tan(l2)


# In[105]:


1/np.sin(l2)


# In[106]:


np.sin(l2)/np.cos(l2)


# In[ ]:


l3=np.array([1,2,3,9])
l4=np.array([4,5,6])
#(1*4)+(2*5)+(3*6)= 4+10+18 =>32

l5=l3*l4
l6=sum(l5)
l6


# In[ ]:


l3=np.array([1,2,3,9])
l4=np.array([4,5,6])

len1=len(l3)
len2=len(l4)

dif=abs(len1-len2)
for i in range(dif):
    l4=np.append(l4,1)
    l5=np.sum(l3*l4)
l5


# ##### 

# In[ ]:





# In[4]:


pip install pandas


# In[5]:


#pandas
import pandas as pd
#pandas is defines as an open library that  provides high performance data manipulation in python 
#data analysis requires lot of processing such as restructuring,cleaning,mergin,manipulating etc..We prefer to perform above
#functionalities coz it is ffast,simple than other tools.
#pandas is built on numpy ,Numpy is required for operating pandas 


# In[6]:


#pandas series 
#pandas series is a data structure with one dimension labelled array
#It is a primary building block of dataframe ,making its rows and columns 


# In[18]:


import numpy as np
labels=['a' ,'b' ,'c']
my_data=[10,20,30]
arr=np.array(my_data)
d={'a':100,'b':200,'c':300}

#syntax -->pandas.series(data=None, index=None ,dtype =None ,name=None ,copy= True or False)


# In[15]:


#syntax -->pandas.Series(data=None,index=None ,dtype =None ,name=None ,copy= True or False)

#eq:
import pandas as pd
pd.Series(my_data)


# In[13]:


type(pd.Series(my_data))


# In[16]:


#series with labels
pd.Series(data=my_data,index=labels)


# In[17]:


pd.Series(data=[print , len ,sum])


# In[20]:


class profile:
    @property
    def display(self):
        print("my name is swa")
s=profile()
s.display


# In[30]:


s1=pd.Series([1,2,3,4],['USA' , 'INDIA', 'CANADA', 'UK'])
s1


# In[25]:


#access val with index
s1[0:3]


# In[31]:


s2=pd.Series([1,2,3,4],['USA','BRAZIL','CANADA','UK'])
s1+s2


# In[34]:


#create a new data in series
s2


# In[38]:


s2['china']="duplicate"
s2


# In[39]:


s2.drop('USA')


# In[40]:


#data frames
#pd.DataFrames(data,row_label,col_label)


# In[42]:


df=pd.DataFrame(np.random.randn(5,4))
df


# In[44]:


#to convert dictionary to DataFrame
d={"col1":[1,2],"col2":[3,4],"col3":[5,6]}
d


# In[50]:


df=pd.DataFrame(np.random.randn(5,4),['a','b','c','d','e'],['w','x','y','z'])
df


# In[45]:


df=pd.DataFrame(data=d)
df


# In[47]:


df=pd.DataFrame(d,['r1','r2'])
df


# In[51]:


df


# In[53]:


#convert rows into coloumns and columns into rows
df.T


# In[54]:


#get all the row names
df.index


# In[55]:


#get all column name
df.columns


# In[56]:


type(df)


# In[57]:


df.dtypesdf.


# In[58]:


df.info()


# In[59]:


df.values


# In[61]:


df.axes


# In[62]:


df.ndim


# In[63]:


df.size


# In[64]:


#to access specific column in df
df


# In[65]:


df['w']


# In[66]:


type(df['w'])


# In[68]:


#to access multiple columns
df[['w','x','y']]


# In[82]:


#to create new column
df['new']=df['w']+df['y']
df


# In[79]:


df.drop('%',axis=1)


# In[84]:


df.drop('new',axis=1)


# In[88]:


df.drop('%',axis=1)


# In[ ]:




