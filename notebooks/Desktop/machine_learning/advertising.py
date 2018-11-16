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

