#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[66]:


columns = [
    'neighbourhood_group', 'room_type', 'latitude', 'longitude',
    'minimum_nights', 'number_of_reviews','reviews_per_month',
    'calculated_host_listings_count', 'availability_365',
    'price'
]

df = pd.read_csv('bnb.csv', usecols=columns)
df.reviews_per_month = df.reviews_per_month.fillna(0)


# In[67]:


df.price


# In[68]:


price_logs=np.log1p(df.price)


# In[69]:


price_logs


# In[70]:


from sklearn.model_selection import train_test_split

df_full_train ,df_test =train_test_split(df, test_size=0.2,random_state =11)
df_train ,df_val =train_test_split(df_full_train, test_size=0.25,random_state =11)



# In[71]:


len(df_train),len(df_test),len(df_val)


# In[106]:



y_train=(df_train.neighbourhood_group == 'Manhattan').astype('int').values
y_test=(df_test.neighbourhood_group == 'Manhattan').astype('int').values
y_val=(df_val.neighbourhood_group == 'Manhattan').astype('int').values


# In[115]:


del df_train['neighbourhood_group']
del df_val['neighbourhood_group']
del df_test['neighbourhood_group']

df_train


# In[124]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text


# In[125]:


train_dicts=df_train.fillna(0).to_dict(orient = 'records')


# In[126]:


dv = DictVectorizer(sparse=False)
x_train=dv.fit_transform(train_dicts)


# In[119]:


dv.get_feature_names()


# In[120]:


len(x_train),x_train.shape


# In[ ]:





# In[121]:


len(y_train),y_train.shape


# In[122]:


dt = DecisionTreeClassifier(max_depth=1)
dt.fit(x_train,y_train1)


# In[127]:


y_pred = dt.predict_proba(x_train)[:,1]
auc=roc_auc_score(y_train,y_pred)
print('train:', auc)


# In[64]:


df.dtypes


# In[130]:


print(export_text(dt , feature_names=dv.get_feature_names()))


# In[27]:


df.neighbourhood_group  


# In[30]:


df.neighbourhood_group.nunique()


# In[100]:


df


# In[134]:


from sklearn.ensemble import RandomForestClassifier


# In[135]:


rf = RandomForestClassifier(n_estimators=10,random_state=1,n_jobs=-1)
rf.fit(x_train,y_train)


# In[137]:


val_dicts=df_val.fillna(0).to_dict(orient='records')
x_val= dv.transform(val_dicts)
y_pred=rf.predict_proba(x_val)[:,1]


# In[144]:


roc_auc_score(y_val,y_pred)


# In[ ]:


scores=[]
for d in [10,15,20,25]:
    for n in range (10,201,10):
        rf = RandomForestClassifier(n_estimators=n,max_depth=d,random_state=1,n_jobs=-1)
        rf.fit(x_train,y_train)
        y_pred=rf.predict_proba(x_val)[:,1]
        auc=roc_auc_score(y_val,y_pred)
        scores.append((d,n,auc))


# In[ ]:


columns=['max_depth','n_estimators','auc']
df_scores=pd.DataFrame(scores,columns=columns)
df_scores


# In[ ]:


#gradient boosting
import xgboost as xgb
features =dv.get_feature_names()

dtrain=xgb.DMatrix(x_train,label=y_train,feature_names=features)
dval=xgb.DMatrix(x_val,label=y_val,feature_names=features)
xgb_params={
    'eta':0.3,
    'max_depth':6,
    'min_child_weight':1,
    
    
    'objective':'binary:logistic',
    'nthreads':8,
    
    'seed':1,
    'verbosity':1,
}
model =xgb.train(xgb_params,dtrain,num_boost_round=10)
y_pred=model.predict(dval)
roc_auc_score(y_val,y_pred)


# In[ ]:


#gradient boosting
import xgboost as xgb
features =dv.get_feature_names()

dtrain=xgb.DMatrix(x_train,label=y_train,feature_names=features)
dval=xgb.DMatrix(x_val,label=y_val,feature_names=features)
xgb_params={
    'eta':0.1,
    'max_depth':6,
    'min_child_weight':1,
    
    
    'objective':'binary:logistic',
    'nthreads':8,
    
    'seed':1,
    'verbosity':1,
}
model =xgb.train(xgb_params,dtrain,num_boost_round=10)
y_pred=model.predict(dval)
roc_auc_score(y_val,y_pred)


# In[ ]:


#gradient boosting
import xgboost as xgb
features =dv.get_feature_names()

dtrain=xgb.DMatrix(x_train,label=y_train,feature_names=features)
dval=xgb.DMatrix(x_val,label=y_val,feature_names=features)
xgb_params={
    'eta':0.01,
    'max_depth':6,
    'min_child_weight':1,
    
    
    'objective':'binary:logistic',
    'nthreads':8,
    
    'seed':1,
    'verbosity':1,
}
model =xgb.train(xgb_params,dtrain,num_boost_round=10)
y_pred=model.predict(dval)
roc_auc_score(y_val,y_pred)

