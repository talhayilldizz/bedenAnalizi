import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from flask import Flask,request,render_template


#flask bağlama
app=Flask(__name__)
#Veriyi Çağır

df=pd.read_excel("Beden_Skalasi.xlsx")
le=LabelEncoder()

df["Giyim Tarzı"]=le.fit_transform(df["Giyim Tarzı"])
df["Göğüs Tipi"]=le.fit_transform(df["Göğüs Tipi"])
df["Beden"]=le.fit_transform(df["Beden"])
print(df)

X=df[["Yaş","Boy (cm)","Kilo (kg)","Giyim Tarzı","Göğüs Tipi"]]
y=df["Beden"]
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,random_state=40)
#düşük oran
# model=LinearRegression()
# model.fit(X_train,y_train)
# y_pred=model.predict(X_test)
# model_score=model.score(X_test,y_test)
# print(f"Tahmin: {model_score*100:.2f}%")

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
rf_model=RandomForestRegressor(n_estimators=70,random_state=1)
rf_model.fit(X_train,y_train)
rf_accurcy=rf_model.score(X_test,y_test)
# print(f"Tahmin2: {rf_accurcy*100:.2f}%")

#flask bağlantısı

@app.route("/",methods=["GET", "POST"])
def index():
    sonuc =None
    if request.method=="POST":
        try:
            yas=int(request.form.get("yas"))
            boy=int(request.form.get("boy"))
            kilo=int(request.form.get("kilo"))
            giyim=request.form.get("giyim")
            gogus=request.form.get("gogus")


            #giyim ve göğüs verileri ayıkla
            if giyim=="bol":
                giyim_kod=0
            elif giyim=="dar":
                giyim_kod=1
            elif giyim=="normal":
                giyim_kod=2
            else:
                raise ValueError("Geçersiz İşlem..")
            
            if gogus=="normal":
                gogus_kod=0
            elif gogus=="siskin":
                gogus_kod=1
            elif gogus=="genis":
                gogus_kod=2
            else:
                raise ValueError("Geçersiz İşlem..")



            # giyim_kod={"bol": 0, "dar": 1, "normal": 2}.get(giyim, -1)
            # gogus_kod={"normal": 0, "siskin": 1}.get(gogus, -1)
            # if giyim_kod== -1 or gogus_kod==-1:
            #     return "Geçersiz İşlem Girdiniz.."

            yeni_veri=pd.DataFrame([[yas,boy,kilo,giyim_kod,gogus_kod]],columns=["Yaş","Boy (cm)","Kilo (kg)","Giyim Tarzı","Göğüs Tipi"])
            yeni_veri_scaled=scaler.transform(yeni_veri)
            tahmin_beden=rf_model.predict(yeni_veri_scaled)
            tahmin_beden=round(tahmin_beden[0])

            beden_map = {0: "L", 1: "M", 2: "S", 3: "XL"}
            sonuc=beden_map.get(tahmin_beden,"Bilinmeyen Beden")
        except Exception as e:
            sonuc= f"Bir Hata Oluştu {str(e)}"
    return render_template("index.html",sonuc=sonuc)

if __name__ == "__main__":
    app.run(debug=True)

