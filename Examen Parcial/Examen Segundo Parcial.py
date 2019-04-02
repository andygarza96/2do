#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MultiLabelBinarizer,LabelEncoder


# In[2]:


data = pd.read_csv("train.csv")


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.head()


# In[6]:


nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False) [:])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)


# In[7]:


bid= data.homepage.unique()
len(bid)
print ("El numero de valores unicos es:")
print (len(bid))


# In[8]:


bid= data.belongs_to_collection.unique()
len(bid)
print ("El numero de valores unicos es:")
print (len(bid))


# In[9]:


bid= data.id.unique()
len(bid)
pbid = (len(bid) * 100) / 3000
print ("El numero de valores unicos es:")
print (len(bid))
print("El Porcentaje de unicos de  es:")
print (pbid)



# In[10]:


bid= data.imdb_id.unique()
len(bid)
pbid = (len(bid) * 100) / 3000
print ("El numero de valores unicos es:")
print (len(bid))
print("El Porcentaje de unicos de  es:")
print (pbid)


# In[ ]:





# In[11]:


target = data.revenue


# In[12]:


data = data.drop(["id"], axis = 1)
data = data.drop(["imdb_id"], axis = 1) 


# In[13]:


data.loc[data["homepage"].notnull(),"homepage"]=1
data["homepage"]=data["homepage"].fillna(0)

data.loc[data["belongs_to_collection"].notnull(),"belongs_to_collection"]=1
data["belongs_to_collection"]=data["belongs_to_collection"].fillna(0)


# In[14]:


data.head()


# In[15]:


mlb = MultiLabelBinarizer()


# In[16]:


def convertStringToList(strVal):
    if type(strVal) is not str:
        return  []
    else:
        return ast.literal_eval(strVal)


# In[17]:


def formatDictColumnAndExtractNames(strVal):
    listOfItems = convertStringToList(strVal)
    return list(map(lambda x: x['name'], listOfItems))


# In[18]:


def extractGenres(data):
    data['genres'] = data['genres'].apply(formatDictColumnAndExtractNames)

    return data.join(pd.DataFrame(mlb.fit_transform(data.pop('genres')),
                          columns=list(map(lambda x: 'genre_'+x,mlb.classes_)),
                          index=data.index))


# In[19]:


data = extractGenres(data)


# In[20]:


data.info()


# In[21]:


data.head()


# In[22]:


data["runtime"]= data["runtime"].fillna(data["runtime"].mean())


# In[23]:


data.loc[data["cast"].notnull(),"cast"]=data.loc[data["cast"].notnull(),"cast"].apply(lambda x : ast.literal_eval(x))
data.loc[data["crew"].notnull(),"crew"]=data.loc[data["crew"].notnull(),"crew"].apply(lambda x : ast.literal_eval(x))


# In[24]:


features_to_fix=["production_companies", "production_countries","Keywords"]

for feature in features_to_fix:
    data.loc[data[feature].notnull(),feature]=    data.loc[data[feature].notnull(),feature].apply(lambda x : ast.literal_eval(x))    .apply(lambda x : [y["name"] for y in x])


# In[25]:


data["cast_len"] = data.loc[data["cast"].notnull(),"cast"].apply(lambda x : len(x))
data["crew_len"] = data.loc[data["crew"].notnull(),"crew"].apply(lambda x : len(x))

data["production_companies_len"]=data.loc[data["production_companies"].notnull(),"production_companies"].apply(lambda x : len(x))

data["production_countries_len"]=data.loc[data["production_countries"].notnull(),"production_countries"].apply(lambda x : len(x))

data["Keywords_len"]=data.loc[data["Keywords"].notnull(),"Keywords"].apply(lambda x : len(x))

data['original_title_letter_count'] = data['original_title'].str.len() 
data['original_title_word_count'] = data['original_title'].str.split().str.len() 
data['title_word_count'] = data['title'].str.split().str.len()
data['overview_word_count'] = data['overview'].str.split().str.len()
data['tagline_word_count'] = data['tagline'].str.split().str.len()


# In[26]:


data["has_tagline"]=1
data.loc[data["tagline"].isnull(),"has_tagline"]=0

data["title_different"]=1
data.loc[data["title"]==data["original_title"],"title_different"]=0

data["isReleased"]=1
data.loc[data["status"]!="Released","isReleased"]=0


# In[27]:


release_date=pd.to_datetime(data["release_date"])
data["release_year"]=release_date.dt.year
data["release_month"]=release_date.dt.month
data["release_day"]=release_date.dt.day
data["release_wd"]=release_date.dt.dayofweek
data["release_quarter"]=release_date.dt.quarter


# In[28]:


data.loc[data["cast"].notnull(),"cast"]=data.loc[data["cast"].notnull(),"cast"].apply(lambda x : [y["name"] for y in x if y["order"]<6]) 


# In[29]:


data["Director"]=[[] for i in range(data.shape[0])]
data["Producer"]=[[] for i in range(data.shape[0])]
data["Executive Producer"]=[[] for i in range(data.shape[0])]

data["Director"]=data.loc[data["crew"].notnull(),"crew"].apply(lambda x : [y["name"] for y in x if y["job"]=="Director"])

data["Producer"]=data.loc[data["crew"].notnull(),"crew"].apply(lambda x : [y["name"] for y in x if y["job"]=="Producer"])

data["Executive Producer"]=data.loc[data["crew"].notnull(),"crew"].apply(lambda x : [y["name"] for y in x if y["job"]=="Executive Producer"])


# In[30]:


data=data.drop(["original_title","overview","poster_path","tagline","status","title",           "spoken_languages","release_date","crew"],axis=1)


# In[31]:


nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False) [:])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)


# In[32]:


to_empty_list=["Keywords","production_companies","production_countries",              "Director","Producer","Executive Producer","cast"]

for feature in to_empty_list:
    data[feature] = data[feature].apply(lambda d: d if isinstance(d, list) else [])


# In[33]:


to_zero=["Keywords_len","production_companies_len","production_countries_len","crew_len","cast_len",
    "tagline_word_count","overview_word_count","title_word_count"]

for feat in to_zero:
    data[feat]=data[feat].fillna(0)


# In[34]:


data['_budget_popularity_ratio'] = data['budget']/data['popularity']
data['_releaseYear_popularity_ratio'] = data['release_year']/data['popularity']
data['_releaseYear_popularity_ratio2'] = data['popularity']/data['release_year']


# In[35]:


data.info()


# In[36]:


to_dummy = ["original_language","production_companies","production_countries",           "Keywords","cast","Director","Producer","Executive Producer"]


# In[37]:


limits=[10,35,15,40,30,15,15,20] 

for i,feat in enumerate(to_dummy):
    mlb = MultiLabelBinarizer()
    s=data[feat]
    x=pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=data.index)
    y=pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=data.index).sum().sort_values(ascending=False)
    rare_entries=y[y<=limits[i]].index
    x=x.drop(rare_entries,axis=1)
    data=data.drop(feat,axis=1)
    data=pd.concat([data, x], axis=1, sort=False)


# In[38]:


data.info()


# In[39]:


list(data.columns)


# In[40]:


data.revenue = np.log1p(data.revenue)
data.budget = np.log1p(data.budget)


# In[41]:


data.budget.describe()


# In[42]:


data.popularity.describe()


# In[43]:


data.runtime.describe()


# In[44]:


data.revenue.describe()


# In[45]:


data2= data[data["budget"]== 0]


# In[46]:


len(data2)


# In[47]:


data2= data[data["budget"]> 18]


# In[48]:


len(data2)


# In[49]:


plt.scatter(x=data['budget'], y=target, color='pink')
plt.ylabel('Revenue')
plt.xlabel('Budget')
plt.show()


# In[50]:


data2= data[data["revenue"]> 19.5]


# In[51]:


len(data2)


# In[52]:


data2= data[data["revenue"]< 4]


# In[53]:


len(data2)


# In[54]:


plt.scatter(x=data['revenue'],y=data['revenue'], color='pink')
plt.xlabel('Revenue')
plt.show()




# In[55]:


data2= data[data["popularity"]<1]


# In[56]:


len(data2)


# In[57]:


data2= data[data["popularity"]> 50]


# In[58]:


len(data2)


# In[59]:


plt.scatter(x=data['popularity'], y=target, color='pink')
plt.ylabel('Revenue')
plt.xlabel('Popularity')
plt.show()


# In[60]:


data2= data[data["runtime"]> 200]


# In[61]:


len(data2)


# In[62]:


data2= data[data["runtime"]< 60]


# In[63]:


len(data2)


# In[64]:


plt.scatter(x=data['runtime'], y=target, color='pink')
plt.ylabel('Revenue')
plt.xlabel('runtime')
plt.show()


# In[65]:


data = data[data['runtime'] >=(60)]
data = data[data['runtime'] <=(200)]


# In[66]:


data = data[data['popularity'] >=(1)]
data = data[data['popularity'] <=(50)]


# In[67]:


data = data[data['revenue'] >=(4)]
data = data[data['revenue'] <=(19.5)]


# In[68]:


data = data[data['budget'] <=(18)]


# In[69]:


data.describe()


# In[70]:


data.head()


# In[71]:


numeric_features = data.select_dtypes(include=[np.number])
print(numeric_features.dtypes)


# In[72]:


corr = numeric_features.corr()

print (corr['revenue'].sort_values(ascending=False)[:10], '\n')


# In[73]:


desired=["budget", "_releaseYear_popularity_ratio2", "popularity","_budget_popularity_ratio","has_tagline","crew_len","United States of America","cast_len","belongs_to_collection"]


# In[74]:


subset = data[desired]


# In[75]:


target=data["revenue"]


# In[76]:


Y = target
X = subset

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=.30)


# In[82]:


modelos = [
    MLPRegressor(hidden_layer_sizes=40, learning_rate_init=0.0002),
    DecisionTreeRegressor(max_depth=7, max_features=3),
    BaggingRegressor(),
    AdaBoostRegressor(n_estimators=8, learning_rate=0.2),
    RandomForestRegressor(max_depth=8, max_features=2),
    linear_model.LinearRegression()
]

for modelo in modelos:
    modelo.fit(X_train, Y_train)
    # score = classifier.score(X_test, Y_test)
    # print(score)
    rmse = mean_squared_error(Y_test, modelo.predict(X_test))
    print("RMSE:")
    print(rmse)


# In[84]:


modelos = [
    MLPRegressor(hidden_layer_sizes=50, learning_rate_init=0.0002),
    DecisionTreeRegressor(max_depth=10, max_features=1),
    BaggingRegressor(),
    AdaBoostRegressor(n_estimators=15, learning_rate=0.5),
    RandomForestRegressor(max_depth=8, max_features=1),
    linear_model.LinearRegression()
]

for modelo in modelos:
    modelo.fit(X_train, Y_train)
    # score = classifier.score(X_test, Y_test)
    # print(score)
    rmse = mean_squared_error(Y_test, modelo.predict(X_test))
    print("RMSE:")
    print(rmse)


# In[86]:


modelos = [
    MLPRegressor(hidden_layer_sizes=30, learning_rate_init=0.0002),
    DecisionTreeRegressor(max_depth=7, max_features=1),
    BaggingRegressor(),
    AdaBoostRegressor(n_estimators=10, learning_rate=0.2),
    RandomForestRegressor(max_depth=7, max_features=1),
    linear_model.LinearRegression()
]

for modelo in modelos:
    modelo.fit(X_train, Y_train)
    # score = classifier.score(X_test, Y_test)
    # print(score)
    rmse = mean_squared_error(Y_test, modelo.predict(X_test))
    print("RMSE:")
    print(rmse)

