import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv("../Veriler/veriler.csv")

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size= 0.33)

sd = StandardScaler()
X_train = sd.fit_transform(x_train)
X_test = sd.transform(x_test)



rdf = RandomForestClassifier(n_estimators=4, criterion='entropy')
rdf.fit(X_train,y_train) # n_estimator yani kac tane agac cizilecegi sayısının belirlenmesinde kullanılan paremetre
y_pred  = rdf.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)