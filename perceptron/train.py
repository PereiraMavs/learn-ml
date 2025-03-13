import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron

s = 'Iris.csv'

print('URL:', s)

df = pd.read_csv(s, header=None, encoding='utf-8')
#print(df)

y = df.iloc[1:101, 5].values
y = np.where(y == 'Iris-setosa', 0, 1)

X = df.iloc[1:101, [1, 3]].values

X = np.array(X, dtype=np.float64)
#print('X:', X)
print('y:', y)

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
#plt.show()

ppn = Perceptron(eta=0.1, n_iter=50)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
