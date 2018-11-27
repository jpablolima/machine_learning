# João Pablo

import pandas as pd
publi = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
#print(publi)
publi.head()

print(publi.shape)

x = publi[['TV', 'radio','newspaper']]
y = publi['sales']

x.shape

y.shape

#aplicação do modelo de regração linear
import seaborn as sns 
sns.pairplot(publi, x_vars=['TV','radio', 'newspaper'], y_vars='sales', size=7, kind='reg')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

#Importação do modelo de regressão linear

from sklearn.linear_model import LinearRegression
reglin = LinearRegression()
reglin.fit(x_train, y_train)

# Cofientes: Aumento nas vendas para cada $1 investido em publicidade
list(zip(['TV', 'radio', 'newspaper'], reglin.coef_))

#Previsão
print(reglin.predict([[230.1,37.8,69.2]]))
y_prev = reglin.predict(x_test)
print(y_prev)

print(y_test)

# Avaliações da Performace
# MAE (Absolute error)

from sklearn import metrics
print(metrics.mean_absolute_error(y_test,y_prev))

# MSE(Mea square error)

print(metrics.mean_squared_error(y_test,y_prev))

# RMSE (Root mean square error)

import numpy as np
print(np.sqrt(metrics.mean_squared_error(y_test,y_prev)))