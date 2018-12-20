import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


diler = [[1,2],[3,44],[5,5],[6,7]]
veriler = pd.read_csv("../Veriler/veriler.csv")
#print(veriler)

#print(veriler[["ulke"]])

ohe = OneHotEncoder(categorical_features="all")

le = LabelEncoder()
dizi = veriler.iloc[:,0:1].values

listeler = veriler.iloc[:,0:1].values

#print(listeler ,"jashdkjasljkdsadas")
dizi[:,0] = le.fit_transform(dizi[:,0])
print(dizi)
#veriler.iloc[:,0:1] = le.fit_transform(dizi[:,0])

dizi = ohe.fit_transform(dizi).toarray()

print(veriler)
print(listeler)
#print(veriler)