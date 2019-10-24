
# coding: utf-8

# In[3]:


# session control
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


# In[5]:


import tensorflow as tf


# In[6]:


matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)


# In[7]:


# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()


# In[8]:


# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)


# In[ ]:


# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-3-session/

