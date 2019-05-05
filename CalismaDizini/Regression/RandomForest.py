import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd



veriler = pd.read_csv("../Veriler/maaslar.csv")

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:3].values

k = x + 0.5
z = x - 0.8

print(k)
print(z)

xDF = veriler.iloc[:,1:2]
yDF = veriler.iloc[:,2:3]

rf_r = RandomForestRegressor(n_estimators=10,random_state=0) # bu apaç yapraklarının dallanışı ile ilgili --->random_state = 0
                                                            # n_estimators=10 10 alt parça için decision tree olustur demek

rf_r.fit(x,y)

plt.scatter(x,y,color="red")
plt.plot(x,rf_r.predict(x),color="green")
plt.plot(x,rf_r.predict(z),color="orange")
plt.plot(x,rf_r.predict(k),color="yellow")
plt.show()





