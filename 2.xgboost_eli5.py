#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

import xgboost as xgb

from tqdm import tqdm

import missingno as msno

import eli5


# In[3]:


df = pd.read_hdf("../input/train_property.h5")
df.shape


# In[4]:


df["Construction phase:_cat"] = df["Construction phase:"].factorize()[0]


# In[5]:


cat_feats = [x for x in df.columns if ":" in x ]

for feat in tqdm(cat_feats):
    df["{}_cat".format(feat)] = df[feat].factorize()[0]

num_feats = [x for x in df.columns if "_cat" in x ]
len(num_feats)


# In[6]:


X = df[num_feats].values
y = df["price"].values
y_log = np.log(y)


# In[7]:


xgb_params = dict(max_depth=5, n_estimators=50, lerning_rate=0.3, random_state=0)
model = xgb.XGBRegressor(**xgb_params)

model.fit(X, y_log)
y_log_pred = model.predict(X)
y_pred = np.exp(y_log_pred)

mean_absolute_error(y, y_pred)


# In[9]:


eli5.show_weights(model, feature_names=num_feats, top=5)


# In[17]:


agg_funcs = [np.mean, np.median, np.std, np.size]
pd.pivot_table(df, index=["Construction phase:"], values=["price"], aggfunc=agg_funcs)


# In[18]:


agg_funcs = [np.mean, np.median, np.std, np.size]
pd.pivot_table(df, index=["Construction phase:", "Housing class:"], values=["price"], aggfunc=agg_funcs)

