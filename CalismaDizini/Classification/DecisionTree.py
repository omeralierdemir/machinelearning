import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
veriler = pd.read_csv("../Veriler/veriler.csv")

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size= 0.33)

sd = StandardScaler()
X_train = sd.fit_transform(x_train) # burada fit_transform önce x_train verilerinden öğren(fit et) sonra uygula(transform et)
X_test = sd.transform(x_test)   #burada da yukarıda öğrendiğin gibi uygula. Muhtemelen train verisi daha fazla olduğu için oradaki öğrenme
                                # x_test için ıygulanacak ogrenmeden daha farklı sonuc ( çünkü daha fazla veri var) oluşturacaktır



dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,y_train)

y_pred  = dt.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)