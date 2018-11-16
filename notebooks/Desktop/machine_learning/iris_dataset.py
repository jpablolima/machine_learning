#João Pablo
#!/usr/bin/env python
# coding: utf-8

# In[44]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[45]:


x = iris.data
print(x)


# In[46]:


y = iris.target
print(y)


# In[47]:



print(iris.target_names)


# In[48]:


print(iris.data.shape)


# In[49]:


print(iris.target.shape)


# ### importação do KNN
# 

# In[50]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)


# ### treinar máquina
# 

# In[51]:


knn.fit(x,y)


# ### Fazer previsões

# In[52]:


species = knn.predict([[5.1, 3.5, 1.4, 0.2]])
print(iris.target_names[species])


# ### Separar dados em dois modelo

# In[54]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
print(x_test.shape)


# ### Avaliação da performace

# In[56]:


knn.fit(x_train, y_train)
previsoes = knn.predict(x_test)
print(previsoes)


# In[58]:


from sklearn import metrics
acertos = metrics.accuracy_score(y_test, previsoes)
print(acertos)


# In[ ]:

###Aplicação do modelo de regressão Logística

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
previsoes_logreg = logreg.predict(x_test)
acertos_logreg = metrics.accuracy_score(y_test, previsoes_logreg)
print(acertos_logreg)



