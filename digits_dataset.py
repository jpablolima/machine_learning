from sklearn import datasets
digits = datasets.load_digits()

print(digits.data.shape)
print(digits.target.shape)

print(digits.data[0])
print('\n')
print(digits.images[0])

import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(2,2))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r)