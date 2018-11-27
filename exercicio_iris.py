from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x, y, test_size=0.25)


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

valores_performace = {}
k = 1

while k <= 25:

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    previsoes = knn.predict(x_test)
    acertos = metrics.accuracy_score(y_test, previsoes)
    valores_performace[k] = round (acertos, 4)
    k += 1
    print(valores_performace)
    import matplotlib.pyplot as plt 


plt.plot(list(valores_performace.keys()),list(valores_performace.values()))

plt.xlabel('valores de k')
plt.ylabel('performace')
plt.show()