#!/usr/bin/env python
# coding: utf-8

# In[1]:


rawUserData=sc.textFile("hdfs://127.0.0.1:9000/ml-100k/u.data")
rawUserData.count()


# In[2]:


rawUserData.first()
#User id, Project id, Rate, Data


# In[6]:


from pyspark.mllib.recommendation import Rating

#get [:3] : User id, Project id, Rate
rawRatings=rawUserData.map(lambda line:line.split("\t")[:3])
rawRatings.take(5)


# In[10]:


#prepare ALS training data
ratingsRDD=rawRatings.map(lambda x:(x[0],x[1],x[2]))
ratingsRDD.take(5)


# In[11]:


numRatings=ratingsRDD.count()
numRatings


# In[12]:


numUsers=ratingsRDD.map(lambda x:x[0]).distinct().count()
numUsers


# In[14]:


numMovies=ratingsRDD.map(lambda x:x[1]).distinct().count()
numMovies


# In[16]:


#traning model
from pyspark.mllib.recommendation import ALS
model=ALS.train(ratingsRDD,10,10,0.01)
print(model)


# In[17]:


model.recommendProducts(100,5)


# In[18]:


model.predict(100,1141)


# In[19]:


model.recommendUsers(product=100,num=5)


# In[20]:


itemsRDD=sc.textFile("hdfs://127.0.0.1:9000/ml-100k/u.item")
itemsRDD.count()


# In[21]:


movieTitle=itemsRDD.map(lambda line:line.split("|")).map(lambda a:(float(a[0]),a[1])).collectAsMap()
len(movieTitle)


# In[23]:


recommendP=model.recommendProducts(100,5)
for p in recommendP:
    print("对用户:"+str(p[0])+",推荐电影: "+str(movieTitle[p[1]])+",推荐评分:"+str(p[2]))


# In[24]:


recommendP=model.recommendProducts(10,5)
for p in recommendP:
    print("对用户:"+str(p[0])+",推荐电影: "+str(movieTitle[p[1]])+",推荐评分:"+str(p[2]))


# In[ ]:




