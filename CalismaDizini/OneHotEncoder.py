from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd


ohe = OneHotEncoder(categorical_features="all")
le = LabelEncoder()
veriler = pd.read_csv('../Veriler/veriler.csv')


ulke = veriler.iloc[:,0:1].values
ulke[:,0] = le.fit_transform(ulke[:,0])

ulke = ohe.fit_transform(ulke).toarray()


print(type(ulke))