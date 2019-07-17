from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import  pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
veriler = pd.read_csv("../Veriler/Churn_Modelling.csv")

le = LabelEncoder()

X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13:].values
X[:,1] = le.fit_transform(X[:,1])

X[:,2] = le.fit_transform(X[:,2])
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()


x_tain, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=0)
ss = StandardScaler()

X_train = ss.fit_transform(x_tain)
X_test = ss.fit_transform(x_test)

classifier = XGBClassifier()

classifier.fit(X_train,y_train)

predict = classifier.predict(x_test)
cm = confusion_matrix(y_test,predict)

print(cm)