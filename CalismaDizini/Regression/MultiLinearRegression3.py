import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

veriler = pd.read_csv("../Veriler/odev_tenis.csv")
# sunny rainy felanı label encoderle label layamayız sebebi ise birbiri ile karşıt veya  ordered bir ilişki bulunmamaktadır. Onun için OneHot yapmalıyız

#play ve windy değerleri lable encoderliyeceğiz

le = LabelEncoder()

play = veriler.iloc[:,-1:].values

play[:,0] = le.fit_transform(play[:,0])
#print(play)
windy = veriler.iloc[:,-2:-1].values
print(windy)

windy[:,0] = le.fit_transform(windy[:,0:1])
#print(windy)


