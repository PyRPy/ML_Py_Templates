
# coding: utf-8

# In[ ]:


# zip, lambda, map


# In[1]:


a=[1,2,3]
b=[4,5,6]
ab=zip(a,b)
print(list(ab))  #需要加list来可视化这个功能


# In[2]:


for i,j in zip(a,b):
     print(i/2,j*2)


# In[ ]:


# lambda 


# In[3]:


fun = lambda x, y : x + y
x = 9
y = -3
print(fun(x, y))


# In[4]:


# map


# In[5]:


def fun(x, y):
    return(x + y)

list(map(fun, [1], [2]))


# In[6]:


list(map(fun, 1, 2))


# In[7]:


list(map(fun, [1,2,3], [4,5,6]))


# In[ ]:


# https://morvanzhou.github.io/tutorials/python-basic/basic/13-03-zip-lambda-map/

