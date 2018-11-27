from sklearn import datasets
digits = datasets.load_digits()

print(digits.data.shape)
print(digits.target.shape)

print(digits.data[0])
print('\n')
print(digits.images[0])

import matplotlib.pyplot as plt
#%matplotlib inline

plt.figure(figsize=(2,2))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r)

from sklearn.model_selection import train_test_split
x=digits.data
y=digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=5)

from sklearn import svm
from sklearn import metrics
classifier = svm.SVC()
classifier.fit(x_train, y_train)
previsoes = classifier.predict(x_test)
acertos = metrics.accuracy_score(y_test, previsoes)
print(acertos)



