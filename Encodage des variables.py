# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 10:07:57 2022

@author: DELL
"""
#%%  Packages
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


#%% *) Chargement de la base de donnée iris.csv
import pandas as pd
data = pd.read_csv('C:/Users/DELL/Desktop/pytho_AS3/Iris .csv')

flower = data['Species']  # endogène  (target)
print(flower.value_counts())

#%% I)Encodage des variable nominales comme notre cas
ECOD = LabelEncoder()
flow_trans = ECOD.fit_transform(flower)
print(flow_trans)

#%%Encodage des variables ordinales
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
flow = encoder.fit_transform(data.Species.values.reshape(-1,1))
flow

#%%Encodage hot encoding
from sklearn.preprocessing import OneHotEncoder
encode = OneHotEncoder()
flow2 = encoder.fit_transform(data.Species.values.reshape(-1,1)).toarray()
flow2
#%%
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
OHE = encoder.fit_transform(data.Species.values.reshape(-1,1)).toarray()
df_OH = pd.DataFrame(OHE, columns = ["Species_" + str(encoder.categories_[0][i]) 
                                     for i in range(len(encoder.categories_[0]))])
df_OH_final = pd.concat([data, df_OH], axis=1)
df_OH_final

#%%MULTIBANIZER BASE
df = pd.DataFrame({"genre": [["action", "drama","fantasy"], ["fantasy","action", "animation"], ["drama", "action"], ["sci-fi", "action"]],
                  "title": ["Twilight", "Alice in Wonderland", "Tenet", "Star Wars"]})
df
#%%MULTIBANIZER
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

res = pd.DataFrame(mlb.fit_transform(df['genre']),
                   columns=mlb.classes_,
                   index=df['genre'].index)
res