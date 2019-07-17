from CalismaDizini.AssociationRuleMining import apyori
import numpy as np
import matplotlib.pyplot as plt
ap = apyori
import pandas as pd

#githup dan uygulama kodu cekip test edilmistir.

veriler = pd.read_csv('../Veriler/sepet.csv', header = None)

t = []
for i in range (0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])


kurallar = ap.apriori(t,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)

print(list(kurallar))
