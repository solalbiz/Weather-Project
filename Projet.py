# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 08:40:38 2020

@author: bizeu
"""

from warnings import simplefilter#LIGNES 8-10: commande pour retirer les Warning lors de l'utilisation de certaines fonctions
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=Warning)
import sklearn as sk#LIGNES 11-19: divers importations
import numpy as np
from matplotlib import style
import pandas as pd
import statistics as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
style.use("ggplot")

df=pd.read_csv("weather.csv",index_col=0)#on importe les données en indiquant que la première colonne (les dates) correspond aux index

plt.hist(df['MinTemp'],bins=30)#on observe l'histogramme des valeurs minimales: on ne voit pas de valeurs aberrantes par exemple
plt.show()
plt.hist(df['MaxTemp'],bins=50)#on observe l'histogramme des valeurs maximales: on remarque des valeurs aberrantes puisque l'axe des abscisses va jusqu'à 100
plt.show()

na=df.isna()#cette commande retourne un DataFrame dans lequel les valeurs sont remplacées par False si la valeur est présente et True si la valeur est manquante
sommecol=pd.DataFrame.sum(na)#cette commande nous permet de visualiser le nombre de valeurs manquantes par colonne
tab=sommecol.to_numpy()#il faut transformer sommecol en tableau numpy pour pouvoir itérer facilement
nom_colonnes=['Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','RISK_MM','RainTomorrow']

for i in range(len(tab)):#on itère sur les colonnes et on enlèves les lignes dont plus d'un tiers des données est manquante
    if tab[i]>142193/3:
        df=pd.DataFrame.drop(df,nom_colonnes[i],axis=1)
        
df=pd.DataFrame.drop(df,'RISK_MM',axis=1)#on supprime RISKMM puisque cette variable est en corrélation directe avec RainTomorrow et rendrait le travail de prédiction trop facile

dfcopy=df
dfcopy=pd.DataFrame.dropna(dfcopy)#on fait une copie de df et on enleve dans dfcopy les lignes avec des valeurs manquantes puisque LabelEncoder ne marche pas avec des valeurs manquantes

le = sk.preprocessing.LabelEncoder()
le.fit(dfcopy['Location'])#on trouve les classes avec la méthode .fit en mettant en argument le tableau numpy de la colonne voulue
list(le.classes_)
RT6=le.transform(dfcopy['Location'])#on transforme le tableau numpy contenant les valeurs qualitatives en tableau numpy contenant les valeurs qualitatives avec la méthode .transform
freq=st.mode(RT6)#on trouve le mode avec la méthode mode
freq2=le.inverse_transform([freq])#on transforme cette valeur pour obtenir la classe la plus fréquente
freq3=' '.join(map(str,freq2))#on la transforme en string
df["Location"].fillna(freq3,inplace=True)#on la remplace dans la colonne de df

le = sk.preprocessing.LabelEncoder()#on transforme df cette fois ci et pas dfcopy : on transforme les valeurs qualitatives en valeurs quantitatives
le.fit(df['Location'])
list(le.classes_)
RT6=le.transform(df['Location'])
df.iloc[:,0]=RT6[:]

le = sk.preprocessing.LabelEncoder()#on recommence la même méthode pour toutes les variables qualitatives
le.fit(dfcopy['RainToday'])
list(le.classes_)
RT1=le.transform(dfcopy['RainToday'])
freq=st.mode(RT1)
freq2=le.inverse_transform([freq])
freq3=' '.join(map(str,freq2))
df["RainToday"].fillna(freq3,inplace=True)

le = sk.preprocessing.LabelEncoder()
le.fit(df['RainToday'])
list(le.classes_)
RT1=le.transform(df['RainToday'])
df.iloc[:,16]=RT1[:]

le = sk.preprocessing.LabelEncoder()#on recommence la même méthode pour toutes les variables qualitatives
le.fit(dfcopy['RainTomorrow'])
list(le.classes_)
RT5=le.transform(dfcopy['RainTomorrow'])
freq=st.mode(RT1)
freq2=le.inverse_transform([freq])
freq3=' '.join(map(str,freq2))
df["RainTomorrow"].fillna(freq3,inplace=True)

le = sk.preprocessing.LabelEncoder()
le.fit(df['RainTomorrow'])
list(le.classes_)
RT5=le.transform(df['RainTomorrow'])
df.iloc[:,17]=RT5[:]

le = sk.preprocessing.LabelEncoder()#on recommence la même méthode pour toutes les variables qualitatives
le.fit(dfcopy['WindGustDir'])
list(le.classes_)
RT2=le.transform(dfcopy['WindGustDir'])
freq=st.mode(RT2)
freq2=le.inverse_transform([freq])
freq3=' '.join(map(str,freq2))
df["WindGustDir"].fillna(freq3,inplace=True)

le = sk.preprocessing.LabelEncoder()
le.fit(df['WindGustDir'])
list(le.classes_)
RT2=le.transform(df['WindGustDir'])
df.iloc[:,4]=RT2[:]

le = sk.preprocessing.LabelEncoder()#on recommence la même méthode pour toutes les variables qualitatives
le.fit(dfcopy['WindDir9am'])
list(le.classes_)
RT3=le.transform(dfcopy['WindDir9am'])
freq=st.mode(RT3)
freq2=le.inverse_transform([freq])
freq3=' '.join(map(str,freq2))
df["WindDir9am"].fillna(freq3,inplace=True)

le = sk.preprocessing.LabelEncoder()
le.fit(df['WindDir9am'])
list(le.classes_)
RT3=le.transform(df['WindDir9am'])
df.iloc[:,6]=RT3[:]

le = sk.preprocessing.LabelEncoder()#on recommence la même méthode pour toutes les variables qualitatives
le.fit(dfcopy['WindDir3pm'])
list(le.classes_)
RT4=le.transform(dfcopy['WindDir3pm'])
freq=st.mode(RT4)
freq2=le.inverse_transform([freq])
freq3=' '.join(map(str,freq2))
df["WindDir3pm"].fillna(freq3,inplace=True)

le = sk.preprocessing.LabelEncoder()
le.fit(df['WindDir3pm'])
list(le.classes_)
RT4=le.transform(df['WindDir3pm'])
df.iloc[:,7]=RT4[:]

nump=df.to_numpy()#on transforme le DataFrame en tableau numpy pour itérer
for i in range(0,142193):#on itère sur la colonne MaxTemp pour transformer toutes les valeurs aberrantes au dessus de 50°C en NaN qu'on va remplacer par la moyenne juste après
    if nump[i][2]>50:
        nump[i][2]='NaN'
df.iloc[:,2]=nump[:,2]

df["MinTemp"].fillna(df["MinTemp"].mean(),inplace=True)#on remplace les valeurs manquantes des variables quantitatives par la moyenne de la colonne
df["MaxTemp"].fillna(df["MaxTemp"].mean(),inplace=True)
df["Rainfall"].fillna(df["Rainfall"].mean(),inplace=True)
df["WindGustSpeed"].fillna(df["WindGustSpeed"].mean(),inplace=True)
df["WindSpeed9am"].fillna(df["WindSpeed9am"].mean(),inplace=True)
df["WindSpeed3pm"].fillna(df["WindSpeed3pm"].mean(),inplace=True)
df["Humidity9am"].fillna(df["Humidity9am"].mean(),inplace=True)
df["Humidity3pm"].fillna(df["Humidity3pm"].mean(),inplace=True)
df["Pressure9am"].fillna(df["Pressure9am"].mean(),inplace=True)
df["Pressure3pm"].fillna(df["Pressure3pm"].mean(),inplace=True)
df["Temp9am"].fillna(df["Temp9am"].mean(),inplace=True)
df["Temp3pm"].fillna(df["Temp3pm"].mean(),inplace=True)


nump=df.to_numpy()#on transforme le DataFrame en tableau numpy pour appliquer StandardScaler

b=np.delete(nump,17,1)#on ne standardise pas RainTomorrow

scaler=StandardScaler().fit(b)#calcule de la moyenne et de l'écart-type
std_nump=scaler.transform(b)#standardisation des valeurs
df.iloc[:,:17]=std_nump#on remplace les valeurs standardisées dans le DataFrame

correlation=df.corr(method='pearson')#on calcule le DataFrame avec les valeurs des coefficients de corrélation
for i in range(0,18):
    print(correlation.ix[:,i])#on itère de façon à ce qu'on imprime les coefficients de corrélation par variable
#on enlève aucune variable malgré le fait que certaines ont un coefficient de corrélation très faible avec la sortie car les garder augmente le accuracy_score de la prédiction
print(df.info())

"""pd.plotting.scatter_matrix(df,figsize=(12,10),diagonal='kde')#on imprime la scatter_matrix correspondant au DataFrame (c'est commenté puisque ce graphe prend beaucoup de temps à se construire)
plt.show()"""


X=np.rint(df.iloc[:,0:16]*1000000)#on transforme les valeurs standardisées décimales en entiers mais sans perdre de précision
print(X)
y = np.rint(df['RainTomorrow'].values*1000000)#de même pour les valeurs de RainTomorrow
print(y)

X_train, X_test, y_train, y_test=sk.model_selection.train_test_split(X,y,random_state=0)#on sépare les sets de test et d'entrainement selon une coupure 75/25 comme préenregistré dans la fonction
clf = LogisticRegression(random_state=0).fit(X_train, y_train)#on calcule les paramètres qui serviront à la prédiction à partir de X_train et y_train
prediction=clf.predict(X_test)#on prédit les valeurs de RainTomorrow à partir des valeurs des autres variables dans X_test 
score1=sk.metrics.accuracy_score(y_test,prediction)#on compare les résultats de la prédiction avec ceux de y_test
print('Accuracy score='+str(score1))#les détails de chaque type d'évaluation sont donnés dans le rapport
score2=sk.metrics.confusion_matrix(y_test,prediction)
print('Confusion matrix='+str(score2))
score3=sk.metrics.precision_score(y_test,prediction,average='macro')#on prend macro car cela nous permet de calculer les métriques pour chaque classes et calculer leur moyenne non pondérée 
print('Precision score='+str(score3))
score4=sk.metrics.recall_score(y_test,prediction,average='macro')
print('Recall score='+str(score4))
score5=sk.metrics.f1_score(y_test,prediction,average='macro')
print('f1 score='+str(score5))

X=X.to_numpy()

kf = sk.model_selection.KFold(n_splits=310)#on sépare les sets de test et d'entrainement avec la méthode des KFold (310 folds car cela optimise le score)
kf.get_n_splits(X)#cela renvoie le nombre d'itérations de fractionnement dans le validateur croisé
for train_index, test_index in kf.split(X):#on itère dans kf.split pour récupérer les sets de test et d'entrainement
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
clf = LogisticRegression(random_state=0).fit(X_train, y_train)#on calcule les paramètres qui serviront à la prédiction à partir de X_train et y_train
prediction=sk.model_selection.cross_val_predict(clf,X_train,y_train)#on compare les résultats de la prédiction avec ceux de y_test
score= sk.model_selection.cross_val_score(clf,X_test,y_test,cv=310)#on calcule le score de chacun des folds, ce qui nous renvoie un np array avec 310 valeurs qui correspondent aux résultats de chacun de folds
print('Cross validation score='+str(np.mean(score)))#on fait une moyenne des cross_val_score pour vérifier l'efficacité de cette méthode
