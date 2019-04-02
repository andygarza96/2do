#!/usr/bin/env python
# coding: utf-8

# ##### Parte 1

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import plotly.plotly as py
from plotly.graph_objs import * 
import plotly.tools as tls


# In[2]:


data = pd.read_json("train.json")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


list(data.columns)


# In[7]:


data.dtypes


# In[8]:


nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False) [:])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)


# In[9]:


data.interest_level.unique()


# In[10]:


tot=49352


# ### Checando los datos 

# In[11]:


bid= data.building_id.unique()
len(bid)
pbid = (len(bid) * 100) / tot
print ("El numero de valores unicos es:")
print (len(bid))
print("El Porcentaje de unicos de Building_id es:")
print (pbid)
if pbid > 30:
    data = data.drop(["building_id"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")
elif (len(bid)) >500 :
    data = data.drop(["building_id"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")


# In[12]:


li= data.listing_id.unique()
len(li)
pli = (len(li) * 100) / tot
print ("El numero de valores unicos es:")
print (len(li))
print("El Porcentaje de unicos de listing_id es:")
print (pli)
if pli > 30:
    data = data.drop(["listing_id"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")
elif (len(li)) >500 :
    data = data.drop(["listing_id"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")


# In[13]:


sa= data.street_address.unique()
len(sa)
psa = (len(sa) * 100) / tot
print ("El numero de valores unicos es:")
print (len(sa))
print("El Porcentaje de unicos de street_address es:")
print (psa)
if psa > 30:
    data = data.drop(["street_address"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")
elif (len(sa)) >500 :
    data = data.drop(["street_address"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")


# In[14]:


c= data.created.unique()
len(c)
print ("El numero de valores unicos es:")
print (len(c))
pc = (len(c) * 100) / tot
print("El Porcentaje de unicos de created es:")
print (pc)
if pc > 30:
    data = data.drop(["created"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")
elif (len(c)) >500 :
    data = data.drop(["created"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")


# In[15]:


d= data.description.unique()
len(d)
print ("El numero de valores unicos es:")
print (len(d))
pdd = (len(d) * 100) / tot
print ("El numero de valores unicos es:")
print (len(d))
print("El Porcentaje de unicos de description es:")
print (pdd)
if pdd > 30:
    data = data.drop(["description"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")
elif (len(d)) >500 :
    data = data.drop(["description"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")


# In[16]:


da= data.display_address.unique()
len(da)
print ("El numero de valores unicos es:")
print (len(da))
pda = (len(da) * 100) / tot
print ("El numero de valores unicos es:")
print (len(da))
print("El Porcentaje de unicos de Display_address  es:")
print (pda)
if pda > 30:
    data = data.drop(["display_address"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")
elif (len(da)) >500 :
    data = data.drop(["display_address"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")


# In[17]:


mid= data.manager_id.unique()
len(mid)
print ("El numero de valores unicos es:")
print (len(mid))
pmid = (len(mid) * 100) / tot
print("El Porcentaje de unicos de manager_id  es:")
print (pmid)
if pmid > 30:
    data = data.drop(["manager_id"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")
elif (len(mid)) >500 :
    data = data.drop(["manager_id"], axis = 1)
    print("Se ha eliminado columna por muchos unicos")


# In[18]:


data = data.drop(["photos"], axis = 1)
data = data.drop(["features"], axis = 1)


# In[19]:


data.head()


# In[20]:


grouped_interest = data.groupby("interest_level")


# In[21]:


grouped_interest.groups


# In[22]:


grouped_interest.get_group("high")


# In[23]:


grouped_interest.get_group("medium")


# In[24]:


grouped_interest.get_group("low")


# In[25]:


dummy_interest = pd.get_dummies(data["interest_level"], prefix="interest")


# In[26]:


dummy_interest.head(10)


# In[27]:


dummy_interest.describe()


# In[28]:


dummy_interest.sum()


# In[29]:


data.dtypes


# In[30]:


numeric_features = data.select_dtypes(include=[np.number])


# In[31]:


colnames = data.columns.values.tolist()
predictors=numeric_features.columns.values.tolist()
target = colnames[2]


# In[32]:


print(target)


# In[33]:


print(predictors)


# In[34]:


data["for_train1"] = np.random.uniform(0,1, len(data))<=0.75


# In[35]:


train, test = data[data["for_train1"]==True], data[data["for_train1"]==False]


# In[36]:


tree = DecisionTreeClassifier(criterion="entropy", min_samples_split=10, random_state=99)


# In[37]:


tree.fit(train[predictors], train[target])


# In[38]:


preds = tree.predict(test[predictors])


# In[39]:


pd.crosstab(test[target], preds, rownames=["Actual"], colnames=["Predictions"])


# In[ ]:





# In[ ]:





# ## Parte 2

# In[40]:


data.describe()


# In[41]:


bathrooms_group = data.groupby(["bathrooms", "interest_level"])
bathrooms_group.describe()


# In[42]:


get_ipython().magic(u'matplotlib inline')
plt.hist(data.bathrooms, bins = 20)
plt.show()


# In[43]:


data = data[data['bathrooms'] >0]
data = data[data['bathrooms'] <=3]


# In[44]:


get_ipython().magic(u'matplotlib inline')
plt.hist(data.bathrooms, bins = 20)
plt.show()


# In[45]:


bathrooms_group = data.groupby(["bathrooms", "interest_level"])
bathrooms_group.describe()


# In[46]:


bedrooms_group = data.groupby(["bedrooms", "interest_level"])
bedrooms_group.sum()


# In[47]:


get_ipython().magic(u'matplotlib inline')
plt.hist(data.bedrooms, bins = 20)
plt.show()


# In[48]:


get_ipython().magic(u'matplotlib inline')
plt.boxplot(data.bedrooms)
plt.show()


# In[49]:


data = data[data['bedrooms'] >=0]
data = data[data['bedrooms'] <=3]


# In[50]:


get_ipython().magic(u'matplotlib inline')
plt.hist(data.bedrooms, bins = 20)
plt.show()


# In[51]:


data.describe()


# In[52]:


data.latitude.describe()


# In[53]:


data.longitude.describe()


# In[54]:


data.price.describe()


# In[55]:


data = data[data['latitude'] >=40.5]
data = data[data['latitude'] <=41]


# In[56]:


data = data[data['longitude'] >=(-74.1)]
data = data[data['longitude'] <=(-73.8)]


# In[57]:


data = data[data['price'] >=(2.470000e+02)]
data = data[data['price'] <=(3.960000e+04)]


# In[58]:


data.describe()


# In[59]:


numeric_features = data.select_dtypes(include=[np.number])


# In[60]:


colnames = data.columns.values.tolist()
predictors=numeric_features.columns.values.tolist()
target = colnames[2]
print(target)
print(predictors)


# In[61]:


data["for_train2"] = np.random.uniform(0,1, len(data))<=0.75
train, test = data[data["for_train2"]==True], data[data["for_train2"]==False]


# In[62]:


tree = DecisionTreeClassifier(criterion="entropy", min_samples_split=28, random_state=99)


# In[63]:


tree.fit(train[predictors], train[target])


# In[64]:


preds = tree.predict(test[predictors])


# In[65]:


pd.crosstab(test[target], preds, rownames=["Actual"], colnames=["Predictions"])


# In[ ]:





# In[ ]:





# In[22]:




