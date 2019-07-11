import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('../Veriler/Ads_CTR_Optimisation.csv')



# UCB
N = 10000  # 10.000 tÄ±klama
d = 10  # toplam 10 ilan var
# Ri(n)
oduller = [0] * d  # ilk basta butun ilanlarÄ±n odulu 0
# Ni(n)
tiklamalar = [0] * d  # o ana kadarki tÄ±klamalar
toplam = 0  # toplam odul
secilenler = []
for n in range(1, N):
    ad = 0  # seÃ§ilen ilan
    max_ucb = 0
    for i in range(0, d):
        if (tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3 / 2 * math.log(n) / tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N * 10
        if max_ucb < ucb:  # max'tan bÃ¼yÃ¼k bir ucb Ã§Ä±ktÄ±
            max_ucb = ucb
            ad = i
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n, ad]  # verilerdeki n. satÄ±r = 1 ise odul 1
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul
print('Toplam Odul:')
print(toplam)

plt.hist(secilenler)
plt.show()
# random tahminde 1000 küsür doğru tahmin var iken şaun 2500 küsür doğru tahmin bulunmakta UCB ile






