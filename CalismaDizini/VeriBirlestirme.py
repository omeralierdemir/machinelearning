import pandas as pd


veriler = pd.read_csv("../Veriler/veriler.csv")

ulke = veriler.iloc[:,0:1].values



yas = veriler.iloc[:,1:4].values


sonuc = pd.DataFrame(data = ulke, index = range(22),columns=["ulke"]) # range(22) çünkü 22 tane verimiz var( satır)

sonuc2 = pd.DataFrame(data=yas,index=range(22), columns=["boy","kilo","yass"])



s = pd.concat([sonuc,sonuc2],axis =1)
print(s)