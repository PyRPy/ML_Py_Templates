
# coding: utf-8

# In[12]:


# place holder
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


# In[13]:


import tensorflow as tf


# In[14]:


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)


# In[15]:


with tf.Session() as sess:
    print(sess.run(output, feed_dict = {input1:[7.], input2:[2.]}))


# In[16]:


# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-5-placeholde/

