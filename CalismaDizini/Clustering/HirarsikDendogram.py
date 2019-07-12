import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


veriler = pd.read_csv('../Veriler/musteriler.csv')

X = veriler.iloc[:,3:].values


ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)

a = X[Y_tahmin==0,0:2]# bu ifade Y_tahmin 0 olanların 0ile 1 sutunlarını a ya ata demek. numpy.array
print(type(X))
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.title('HC')
plt.show()

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()



# Dendograma bakınca en iyi 2 ve 4 kümeye bölmenin en iyi performansı vereceği söylenebilir.












