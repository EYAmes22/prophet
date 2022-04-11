#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
from prophet import Prophet


# In[3]:


df = pd.read_excel('OLAF.xlsx') 
df


# In[4]:


df.isnull().sum()


# In[5]:


s = pd.Series(df['QuantitÃ©'])
d=s.interpolate()
d


# In[6]:


d.isnull().sum()
df['QuantitÃ©']=d
df.isnull().sum() 


# In[7]:


df.set_index('Date', inplace = True)
df


# In[8]:


import matplotlib.pyplot as plt 

P=df.groupby(['Date'],as_index='False').agg({'QuantitÃ©':'sum'})
P
plt.figure(figsize=(16,8))
plt.plot(P)
plt.show


# In[9]:


P


# In[10]:


P.reset_index(inplace=True)


# In[11]:


P = P.rename(columns={"Date": "ds",'QuantitÃ©':'y'}, inplace=False)
P


# In[12]:


P['ds']=pd.DatetimeIndex(P['ds'])
m=Prophet()
m.fit(P)


# In[13]:


P


# In[14]:


P.dtypes


# In[15]:


future=m.make_future_dataframe(periods=365)
future.tail()


# In[16]:


forecast=m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[17]:


fig1=m.plot(forecast)


# In[18]:


fig2=m.plot_components(forecast)


# In[19]:


from prophet.plot import plot_plotly,plot_components_plotly
plot_plotly(m,forecast)


# In[20]:


plot_components_plotly(m,forecast)


# In[21]:


y=P['y']
yhat=forecast['yhat'].head(91)
SSE=sum((yhat-y)**2)


# In[22]:


SSE


# In[23]:


from sklearn.metrics import r2_score
r2_score(y, yhat)


# In[ ]:




