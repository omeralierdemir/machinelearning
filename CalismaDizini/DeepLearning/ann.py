import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv("../Veriler/Churn_Modelling.csv")

le = LabelEncoder()

X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13:].values
X[:,1] = le.fit_transform(X[:,1])

X[:,2] = le.fit_transform(X[:,2])
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()


x_tain, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=0)
ss = StandardScaler()

X_train = ss.fit_transform(x_tain)
X_test = ss.fit_transform(x_test)

classifier = Sequential()
classifier.add(Dense(6,init="uniform",input_dim=11,activation="relu"))
classifier.add(Dense(6,init="uniform",activation="relu"))
classifier.add(Dense(1,init="uniform",activation="sigmoid"))

classifier.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(X_train,y_train,epochs=50)

y_pred = classifier.predict(X_test)
y_pred = (y_pred >0.5)
cm = confusion_matrix(y_test,y_pred)
print(cm)
"""
hi = tf.constant("\n merhaba")
sess = tf.Session()
print(sess.run(hi))

"""
