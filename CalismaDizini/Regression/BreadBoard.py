import pandas as pd


veriler = pd.read_csv("../Veriler/satislar.csv")


aylar = veriler["Aylar"].values
print(aylar)