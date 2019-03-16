import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd



veriler = pd.read_csv("../Veriler/maaslar.csv")

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:3].values

xDF = veriler.iloc[:,1:2]
yDF = veriler.iloc[:,2:3]

pol_reg = PolynomialFeatures(degree=2)  # bu bizim ön işlememiz biz herhangi bir sayıyı bu sayede polinomal bir şekilde
                                        # ifade edebiliriyoruz. Kütüphaneye dikkat edersen sklearn.propossing den import edilior yani ön işleme
x_pol_reg = pol_reg.fit_transform(x)

print(x_pol_reg)
print(y)

# buradaki amacımız öğrenme verilerimizi polinomal veriye çevirip sonra linner modele vermek bu sayede daha başarılı sonuç elde edilmektedir.
# y = b0 + b1*x + b2*x^2 + b3*x^3 .... denklemindeki b katsayılarını y verilerine göre öğrenmek sonrasında tahminde bulunabilmektir.
lin_reg = LinearRegression()
lin_reg.fit(x_pol_reg,y)

plt.scatter(x,y,edgecolors="red")
plt.plot(x,lin_reg.predict(pol_reg.fit_transform(x)),color="blue")
plt.show()

pol_reg2 = PolynomialFeatures(degree=4) # derece arttırldığında tahmine daha çok yaklaşılmıştır.
x_pol_reg2 = pol_reg2.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_pol_reg2,y)

plt.scatter(x,y,edgecolors="green")
plt.plot(x,lin_reg2.predict(pol_reg2.fit_transform(x)),color="blue")
plt.show()