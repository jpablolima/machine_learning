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

# Leitura de imagem de um dígito

import numpy as  np
import matplotlib.image as mpimg

img = mpimg.imread('number.png')
#print(img)


def rgb2gray(rgb):
    img_array = np.dot(rgb[...:3], [0.299,0.587,0.114])
    #print(img_array)
    img_array = (16 - (img_array *16)).astype(int)
    #print(img_array)
    img_array = img_array.flatten()
    #print(img_array)
    return img_array

    
    rgb2gray(img)    


