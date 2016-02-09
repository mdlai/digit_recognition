import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

#Preprocessing
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu
from skimage.transform import resize

df = pd.read_csv('train.csv')

images = df[df.columns[1:]]
given_label = df['label']
boximages = np.array(images).reshape(42000,28,28)

def preprocess(num):
    thresh = threshold_otsu(num)
    binary = num > thresh
    binary = binary.astype(int)
    return resize(regionprops(binary)[0].image.astype(float),(28,28))

resized = np.empty([42000,28,28])
for i, pic in enumerate(boximages):
    resized[i] = preprocess(pic)

X_train, X_test, y_train, y_test = train_test_split(images,
                                                    given_label,
                                                    test_size = .30,random_state = 4444)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)

y_fit = knn.predict(X_test)

print accuracy_score(y_test,y_fit)


flatten = resized.reshape(42000, 784)
X_train, X_test, y_train, y_test = train_test_split(flatten,
                                                    given_label,
                                                    test_size = .30, random_state = 4444)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)

y_fit = knn.predict(X_test)

print accuracy_score(y_test,y_fit)
