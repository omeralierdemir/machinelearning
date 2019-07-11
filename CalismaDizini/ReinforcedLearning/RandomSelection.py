import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

veriler = pd.read_csv("../Veriler/Ads_CTR_Optimisation.csv")

N = 10000
random_length = 10
reward = []
odul = 0
for i in range(N):

    tahmin = random.randrange(random_length)

    etki = veriler.values[i,tahmin]
    odul = odul + etki
    reward.append(tahmin)

plt.hist(reward)
plt.show()

print(odul)

#Bu uygulamada 10 adet reklamın tıklanma verisini tutmakta veriler değişkeni.
# random olarak bizde tahminde bulunarak tahmin edilen reklamın yayınlıyoruz.
#Eğer bu reklam tıklanmışsa reward(ödüle) 1 ekliyo, tıklanmadıysa 0 ekliyoruz.
#Random olarak tahminde bulunduğundan 10000 tahminde yaklaşık 1000 doğru tahmin
#Herhangi bir pekiştirmeli öğrenmede bu sayı çok üst seviyek
#Burada herhangi bir öğreme veya zeka işlemi bulunmamktadır.
# Amaç Upper Confidence Bound yapısına neden ihtiyaç olduğunu anlamaktır.
