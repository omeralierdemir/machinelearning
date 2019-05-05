import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd



veriler = pd.read_csv("../Veriler/maaslar.csv")

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:3].values

k = x + 0.5
z = x - 0.4

print(k)
print(z)

xDF = veriler.iloc[:,1:2]
yDF = veriler.iloc[:,2:3]

d_tr = DecisionTreeRegressor(random_state=0) # bu apaç yapraklarının dallanışı ile ilgili

d_tr.fit(x,y)

plt.scatter(x,y,color="red")
plt.plot(x,d_tr.predict(x),color="green") # bu şu demek oluyor şimdi her ayırdığı bölgenin  ortalamasını alıp ordaki veriyi veriyo
plt.plot(x,d_tr.predict(z),color="orange") # yani bir birine belirli yakınlıktaki değerlerde hep aynı sonucu üretecektir. buyüzden
plt.plot(x,d_tr.predict(k),color="orange")# bütün plotlar üstüste geldi.
plt.show()



