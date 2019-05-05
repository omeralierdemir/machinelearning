import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
#import statsmodels.formula.api as sm

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
#print(type(ulkeDFrame))
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

#print(y_test2)
#print(boy_pred)



#-----------------backwardElimination----------------------------


X = np.append(arr=np.ones((22,1)).astype(int),values=newData,axis=1) # birebirlik array oluşturuldu int türünde, newData arrayine yukardan aşağı şeklinde eklendi
#print(X)
X_l = newData.iloc[:,[0,1,2,3,4,5]].values  # daha sonradan üzerinde oynma yapabilmek için bir dizi şeklinde aldık
print(type (X_l))  # burada yapılan asıl olay tam olarak bizim multilinner regrestion modelimizde y = B + B1X1 + B2X2 +  B3X3 ... bir denklem var. Bu denklem
                # için elimizde bağımsız değişkenlerimiz bulunmakta ama ilgi B sabiti yok bu sabiti de ekleyebilmek için 1 lerden oluşan bir
                # sutun ekledik. Bu sutun 1 lerden oluşma sebebi katsayısını 1 olmasıdır.

boyArray = boyDFrame.iloc[:,0:1].values
result_OLS = sm.OLS(endog=boyArray,exog=X_l) # boy verisine göre diğer değişkenlerin bilgilerini "Ols raporu"(koveryans varyans p_value vb.) çıkarıyor. Bu çıkarma işleminin
r = result_OLS.fit()                        # gerçekleşmesi için fit() demen lazım.

print(r.summary()) # çıkarılan değerlerin özeti. Buradan P_value olan en büyük değeri elemeliyiz.

X_l2 = newData.iloc[:,[1,2,3,4,5]].values
result_OLS2 = sm.OLS(endog=boyArray,exog=X_l2) # burada 0. index deki değeri eledik. Bu şekilde devam etcek. Genelde 0.05 altında olana kadar eleme işlemi devam eder

r2 = result_OLS2.fit()

print(r2.summary())