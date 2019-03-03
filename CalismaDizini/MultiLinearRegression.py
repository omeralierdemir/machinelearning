import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

le = LabelEncoder()

ohe = OneHotEncoder(categories="auto")
veriler = pd.read_csv("../Veriler/veriler.csv")

ulke = veriler.iloc[:,0:1].values

diger = veriler.iloc[:,1:4].values
ulke[:,0] = le.fit_transform(ulke[:,0])# label encodere çevirdir bu sayede etiketledik 0-1 ile
ulke = ohe.fit_transform(ulke).toarray()
#print(type(ulke))
#print(ulke)





cinsiyet = veriler.iloc[:,4:5].values


cinsiyet[:,0] = le.fit_transform(cinsiyet[:,0])# label encodere çevirdir bu sayede etiketledik 0-1 ile
cinsiyet = ohe.fit_transform(cinsiyet).toarray()
#print(type(ulke))
#print(cinsiyet)

ulkeDFrame = pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
digerDFrame = pd.DataFrame(data=diger,index=range(22),columns=["boy","kilo","yas"])
cinsiyetDFrame = pd.DataFrame(data=cinsiyet[:,0:1],columns=["cinsiyet"])# burada dummy trap a düşmemek için teki seçildi. direk label encoder almanda yeterli
                                                                        #olacaktır.
print(type(ulkeDFrame))
#print(digerDFrame)
#print(cinsiyetDFrame)

newXData = pd.concat([ulkeDFrame,digerDFrame],axis=1)
newData = pd.concat([ulkeDFrame,digerDFrame,cinsiyetDFrame],axis=1)





#---------------------buraya kadar veri ön isleme asamasıdır-----------------------------

regression = LinearRegression()

x_train,x_test,y_train,y_test = train_test_split(newXData,cinsiyetDFrame,test_size=0.33,random_state=0)  # cinsiyet tahmini için

regression.fit(x_train,y_train)

cin_pred = regression.predict(x_test)
#print(y_test)
#print(cin_pred)

#--------------burada boy için veri ön işleme yapıldı daha doğrusu uygun dataframe değişkenleri elde edildi -----------

boyDFrame = newData.iloc[:,3:4]
sagDFrame = newData.iloc[:,0:3]
solDFrame = newData.iloc[:,4:]

boyTrain = pd.concat([sagDFrame,solDFrame],axis=1)

x_train2,x_test2,y_train2,y_test2 = train_test_split(boyTrain,boyDFrame,test_size=0.33,random_state=0) # boy tahmini için

#print(x_train2)

#--------boy tahmin etme olayı-------

regression.fit(x_train2,y_train2)

boy_pred = regression.predict(x_test2)

print(y_test2)
print(boy_pred)