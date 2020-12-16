import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("kaggleCompetition.csv")
data = data.values

x = data[0:1460, :-1]
y = data[0:1460, -1] 
x_comp = data[1460:,:-1] 
y_comp = data[1460:,-1]

scaler = preprocessing.StandardScaler().fit(x) 
x = scaler.transform(x)
x_comp = scaler.transform(x_comp)


print(x_comp)
print(y_comp)
