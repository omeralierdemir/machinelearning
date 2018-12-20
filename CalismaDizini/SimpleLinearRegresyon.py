

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

veriler = pd.read_csv('../Veriler/satislar.csv')

#print(veriler)

#print(veriler)

aylar = veriler[["Aylar"]]

satislar = veriler[["Satislar"]]

print(aylar)
print(satislar)

"""
aylar = veriler.iloc[:,0:1].values
satislar = veriler.iloc[:,1:2].values
"""
x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#print(x_test,x_train,y_test,y_train)

print(X_test)
print(X_train)