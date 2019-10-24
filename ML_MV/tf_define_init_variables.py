
# coding: utf-8

# In[9]:


# define variables
# session control
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import tensorflow as tf


# In[3]:


state = tf.Variable(0, name = 'counter')
print(state.name)


# In[4]:


one = tf.constant(1)


# In[5]:


new_value = tf.add(state, one)


# In[12]:


update = tf.assign(state, new_value)


# In[13]:


init = tf.initialize_all_variables() # don't forget


# In[14]:


with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


# In[15]:


# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-4-variable/

