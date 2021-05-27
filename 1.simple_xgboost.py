#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

import xgboost as xgb

from tqdm import tqdm

import missingno as msno


# In[71]:


df = pd.read_hdf("../input/train_property.h5")
df.shape


# In[72]:


msno.matrix(df.sample(250));


# In[73]:


msno.bar(df.sample(250));


# In[74]:


msno.heatmap(df.sample(250));


# In[10]:


df.sample(5)


# In[76]:


df.info()


# In[13]:


df["Construction phase:"].value_counts()


# In[14]:


df["Construction phase:"].value_counts(normalize=True)


# In[25]:


construction_phase_dict = {val:idx for idx, val in enumerate(df["Construction phase:"].unique())}
construction_phase_dict


# In[22]:


df["Construction phase:"].head(10)


# In[26]:


df["Construction phase:"].map(construction_phase_dict).head(10)


# In[77]:


ids, labels = df["Construction phase:"].factorize()


# In[78]:


len(ids), labels


# In[32]:


df.shape


# In[36]:


labels[5]


# In[79]:


df["Construction phase:_cat"] = df["Construction phase:"].factorize()[0]


# In[80]:


df["Construction phase:_cat"].value_counts()


# In[81]:


labels


# In[82]:


cat_feats = [x for x in df.columns if ":" in x ]
cat_feats


# In[83]:


for feat in tqdm(cat_feats):
    df["{}_cat".format(feat)] = df[feat].factorize()[0]


# In[51]:


df.sample(1)


# In[84]:


num_feats = [x for x in df.columns if "_cat" in x ]
len(num_feats)


# In[85]:


X = df[num_feats].values
y = df["price"].values


# In[86]:


xgb_params = dict(max_depth=5, n_estimators=50, lerning_rate=0.3, random_state=0)
model = xgb.XGBRegressor(**xgb_params)


# In[87]:


model.fit(X, y)


# In[89]:


y_pred = model.predict(X)


# In[90]:


mean_absolute_error(y, y_pred)


# In[91]:



import sklearn
sklearn.metrics.SCORERS.keys()


# In[98]:


from sklearn.model_selection import cross_val_score


# In[103]:


scores = cross_val_score(model, X, y, cv=3, scoring="neg_median_absolute_error")


# In[105]:


np.mean(scores), np.std(scores)


# In[106]:


df["price"].hist();


# In[109]:


np.log(df["price"]).hist(bins=100);


# In[110]:


np.log(100)


# In[111]:


np.exp(np.log(100))


# In[112]:


xgb_params = dict(max_depth=5, n_estimators=50, lerning_rate=0.3, random_state=0)
model = xgb.XGBRegressor(**xgb_params)

y_log = np.log(y)

model.fit(X, y_log)
y_log_pred = model.predict(X)
y_pred = np.exp(y_log_pred)

mean_absolute_error(y, y_pred)


# In[ ]:




