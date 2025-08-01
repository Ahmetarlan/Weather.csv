import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("C:\\Users\\knox0\\Desktop\\Summary of Weather.csv\\Summary of Weather.csv", na_values=['#VALUE!'])
df = pd.DataFrame(df)

print(df.head())
print("----------- \n")
print(df.isnull().sum())
## Kolon temizliği yapıyoruz
df = df.drop(columns=["DR","SPD","SND","FT","FB","FTI","ITH", "PGT","SD3","RHX","RVG","WTE","WindGustSpd","RHN","PoorWeather","TSHDSBRSGF","Snowfall"])
print(df.isnull().sum())


## Hata veren kolonun sınıfına bakıyoruz
print(type(df["PRCP"]))
print(df["PRCP"].describe())

df["Date"] = pd.to_datetime(df["Date"])

df["PRCP"] = df["PRCP"].replace('T', 0)
df["PRCP"] = pd.to_numeric(df["PRCP"], errors='coerce')
df["PRCP"] = df["PRCP"].fillna(df["PRCP"].mean())
print(df["PRCP"].isna().sum())

df["Precip"] = df["Precip"].replace("T",0.54)
df["Precip"] = df["Precip"].astype(float)

df["SNF"] = df["SNF"].replace("T", 0)
df["SNF"] = pd.to_numeric(df["SNF"], errors="coerce")
df["SNF"] = df["SNF"].fillna(df["SNF"].mean())
print(df["SNF"].isna().sum())


df["MAX"] = df["MAX"].fillna(df["MAX"].mean())
df["MIN"] = df["MIN"].fillna(df["MIN"].mean())
df["MEA"] = df["MEA"].fillna(df["MEA"].mean())


plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')  
plt.title("Korelasyon Isı Haritası")
plt.show()



X = df.drop(columns=["MeanTemp", "Date"])
y = df["MeanTemp"]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.25, random_state=15)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression(n_jobs=1)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Ortalama Kare Hatası (MSE):", mse)
print("R² Skoru:", r2)

plt.scatter(y_test, y_pred)
plt.xlabel('Gerçek Ortalama Sıcaklık')
plt.ylabel('Tahmin Edilen Ortalama Sıcaklık')
plt.title('Gerçek vs Tahmin Edilen Ortalama Sıcaklık')
plt.grid(True)
plt.show()
