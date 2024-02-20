#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy
t1=(1,2,3,4)
arr=numpy.array(t1)


# In[5]:


arr


# In[1]:


#to add elements to array
import numpy as np
new_arr=np.append(np_ar,78,789)
new_arr


# In[ ]:


#to add multiple values to array
new_ar= np.append(np_ar,[56,67,78])
new_ar


# In[12]:


ar2=[1,2,3,4]


# In[16]:


import numpy as np
np_ar=np.array(ar2)
np_ar


# In[18]:


new_ar= np.append(np_ar,[56,67,78])
new_ar


# In[19]:


#delete ele from array
np_ar


# In[20]:


nr=np.delete(np_ar,np.where(np_ar==2.0))
nr


# In[25]:


#delete no of values
nr=np.setdiff1d(np_ar,[1.0,3])
nr


# In[26]:


#delete val based on index
nr=np.delete(np_ar,[1,2])
nr


# In[30]:


l1=[1,45,23,90,78,56,35,12,15]
new_ar=np.array(l1)


# In[28]:


#filter in numpy
new_ar<30


# In[29]:


new_ar[new_ar<30]


# In[34]:


#print the values less than 20 greater than 50
#for and operation using &
#for or operation use |
(new_ar>20 )&( new_ar<50)


# In[35]:


new_ar[(new_ar>20 ) & (new_ar<50)]


# In[39]:


#to replace val in array
new_ar[new_ar==23]=156


# In[40]:


new_ar


# In[41]:


#to create array full of zeroes
np.zeros(5)


# In[42]:


#to create array 1d full of zeroes
np.zeros(5)


# In[45]:


#to create array 2d full of zeroes
#syntax-->([bo.of rows,no ofcoloumns])
np.zeros([3,4])


# In[47]:


#to create array 1d full of ones
np.ones(5)


# In[60]:


#to create array 2d full of ones
n=np.ones([3,5],dtype=int)
n


# In[61]:


#type convert the elements in array
#to find the numpy data type of element present in array-->dtype
n.dtype


# In[63]:


#we can convert the values in numpy array to various bits
#they are
#int8,int16,int32,int64
#float16,float32,float64
#complex64,complex128
ar=np.zeros([2,4],dtype='int64')
ar


# In[65]:


ar=np.zeros([2,4],dtype='float16')
ar


# In[66]:


ar=np.zeros([2,4],dtype='complex64')
ar


# In[67]:


np.int16(ar)


# In[70]:


#how to create range of elements as array
#how to convert a 1d array to multidimensional array
#how create array pf identity matrix
#how to perfom slicing in multidimensional array
#how to find the mean,median,sum,variance,standard deviation of an array



# In[7]:


import numpy as np
n=np.arange(4,10)
n


# In[8]:


n=np.arange(4,10,2)
n


# In[ ]:




