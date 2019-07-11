import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("../Veriler/Ads_CTR_Optimisation.csv")


# UCB
N = 10000  # 10.000 tÄ±klama
d = 10  # toplam 10 ilan var
# Ri(
# Ni(n)
toplam = 0  # toplam odul
secilenler = []
birler = [0] * d
sifirlar = [0] * d
for n in range(1, N):
    ad = 0  # seÃ§ilen ilan
    max_th = 0
    for i in range(0, d):
        rasbeta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n, ad]  # verilerdeki n. satÄ±r = 1 ise odul 1
    if odul == 1:
        birler[ad] = birler[ad] + 1
    else:
        sifirlar[ad] = sifirlar[ad] + 1
    toplam = toplam + odul
print('Toplam Odul:')
print(toplam)

plt.hist(secilenler)
plt.show()
# ucb ye göre daha başarılı 2700 e yakın başarılı tahmin var. Ancak gerçek hayatta belirli bir dağılıma sahip
# sistemler üzerinde çalışmak zor






