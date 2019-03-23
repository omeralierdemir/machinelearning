import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd



veriler = pd.read_csv("../Veriler/maaslar.csv")

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:3].values

xDF = veriler.iloc[:,1:2]
yDF = veriler.iloc[:,2:3]

s_scaler = StandardScaler()

x_olcekli = s_scaler.fit_transform(x)
y_olcekli = s_scaler.fit_transform(y)


svr = SVR(kernel="rbf") #rbf ,sigmoid ,linear, poly, precomputed  şeklinde deniyebilirsin.

svr.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color="blue")
plt.plot(x_olcekli,svr.predict(x_olcekli),color="red")
plt.show()


#print(svr.predict(6.6))