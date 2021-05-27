#!/usr/bin/env python
# coding: utf-8

# In[1]:


ls ../input/


# In[54]:


import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor

pd.set_option('display.max_columns', 500)


# In[3]:


df = pd.read_hdf("../input/train_property.h5")
df.shape


# In[5]:


df.head()


# In[10]:


df.sample(5)


# In[8]:


df.columns


# In[12]:


df["price"].sample(5)


# In[13]:


df["price"].max


# In[22]:


lista = [1,3,7,11,2,-6,0]


# In[28]:


najmniejsza = None
najwieksza = None

for i in lista:

if najmniejsza == None or najmniejsza > i: 
        najmniejsza = i
        
    if najwieksza == None or najwieksza < i:
        najwieksza = i
        
print ("najmniejsza liczba to:", najmniejsza)
print ("największa liczba to:", najwieksza)


# In[29]:


max("price")


# In[30]:


max(price)


# In[31]:


df["price"].describe()


# In[34]:


df["price"].hist(bins=100);


# In[37]:


np.log10(df["price"]).hist(bins=100);


# In[38]:


np.log10(10), np.log10(100), np.log10(1000)


# In[39]:


mean_price = df["price"].mean()


# In[40]:


mean_price


# In[46]:


df["price_pred"] = mean_price


# In[47]:


df["error_pred"] = np.abs(df["price_pred"] - df["price"])
df["error_pred"].mean()


# In[48]:


mean_absolute_error(df["price"], df["price_pred"])


# In[50]:


median_price = df["price"].median()


# In[51]:


df["price_pred_median"] = median_price


# In[52]:


mean_absolute_error(df["price"], df["price_pred_median"])


# In[53]:


df["price"].median()


# In[57]:


X = df.values
y = df["price"].values

X.shape, y.shape


# In[62]:



def run_model(model, X, y):

    model.fit(X,y)
    y_pred = model.predict(X)

    return mean_absolute_error (y, y_pred)


# In[63]:


model = DummyRegressor(strategy="mean")

run_model(model, X, y)


# In[64]:


model = DummyRegressor(strategy="median")

run_model(model, X, y)


# In[ ]:




