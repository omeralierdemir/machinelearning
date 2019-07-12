
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('../Veriler/musteriler.csv')

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 42)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

print(sonuclar)
plt.plot(sonuclar)
plt.show()

#WCSS değerlerine bakınca grafikte yani clustering başarılarına göre başaroların grafiğine bakınca kırılma noktası 4 kümeye bölmede kırıldığı görülmekte
#Bu veriler üzerinde 4 kümeye ayırma işlemi daha iyi sonuç verecektir.















