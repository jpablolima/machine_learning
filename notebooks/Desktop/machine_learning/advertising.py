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

list(zip(['TV', 'radio', 'newspaper'], reglin.coef_))