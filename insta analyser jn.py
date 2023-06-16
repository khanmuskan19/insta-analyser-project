#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
# import os
# print(os.getcwd())
data = pd.read_csv(r"C:\Users\Muskan Khan\OneDrive\Documents\Instagram data.csv", encoding = 'latin1')
print(data.head())
data.isnull().sum()
data=data.dropna()
data.info()
plt.figure(figsize=(18,18))
plt.style.use("fivethirtyeight")
plt.title("Distribution of Impression from Home")
sns.displot(data['From Home'])
plt.show


# In[5]:


plt.figure(figsize=(10,8))
plt.title("Distribution of Impressions from Hashtags")
sns.distplot(data['From Hashtags'])
plt.show


# In[11]:


plt.figure(figsize=(10,8))
plt.title("Distribution of Impression from Explore")
sns.distplot(data['From Explore'])


# In[9]:


home=data['From Home'].sum()
hashtags=data['From Hashtags'].sum()
explore=data['From Explore'].sum()
other=data['From Other'].sum()


labels=['From Home','From Hashtags','From Explore','Other']
values=[home,hashtags,explore,other]
colors = sns.color_palette('pastel')[0:5]
explode = [0.1, 0.1, 0.1, 0.1]

plt.pie(values,labels=labels,colors=colors,explode=explode, autopct='%.0f%%')
plt.title("Impressions on Instagram Posts from Various resources")
plt.show



# In[7]:


home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig=px.pie(data,values=values,names=labels, title='Impression on Instagram Posts from Various Sources',hole=0.5)
fig.show()


# In[3]:


text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()


# In[17]:


pip install --upgrade pip


# In[2]:


pip install Pillow


# In[3]:


pip install --upgrade Pillow


# In[10]:



figure=px.scatter(data_frame=data, x="Impressions",y='Likes',size="Likes",trendline='ols',title="Relationship between Likes and Impressions")
figure.show()


# In[4]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Total Impressions")
figure.show()


# In[5]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title = "Relationship Between Shares and Total Impressions")
figure.show()


# In[6]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Comments", trendline="ols", 
                    title = "Relationship Between Saves and Total Impressions")
figure.show()


# In[10]:


correlation=data.corr()
print(correlation["Impressions"].sort_values(ascending=False))


# In[11]:


conversion_rate=(data["Follows"].sum()/data["Profile Visits"].sum())*100
print(conversion_rate)


# In[12]:


figure=px.scatter(data_frame=data, x="Profile Visits",y="Follows",trendline="ols",title="Relationship Between Profile Visits and Followers Gained")
figure.show()


# In[9]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
data = pd.read_csv(r"C:\Users\Muskan Khan\OneDrive\Documents\Instagram data.csv", encoding = 'latin1')
x=np.array(data[['Likes','Saves','Comments','Shares','Profile Visits','Follows']])
y=np.array(data["Impressions"])
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
model=PassiveAggresiveRegressor()
model.fit(xtrain,ytrain)
model.score(xtest,ytest)


# In[ ]:




