
# coding: utf-8

# In[6]:


# tensor flow example
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


# In[3]:


import tensorflow as tf
import numpy as np


# In[8]:


# data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3


# In[10]:


plt.scatter(x = x_data, y = y_data)
plt.show()


# In[11]:


# construct model
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases


# In[12]:


# calculate 'loss' or 'error'
loss = tf.reduce_mean(tf.square(y - y_data))


# In[13]:


# optimization
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


# In[14]:


# train
init = tf.global_variables_initializer()


# In[16]:


# iterate
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))


# In[ ]:


# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-2-example2/

