import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

df=pd.read_csv("gender_classification_v7.csv")
x_data=df.drop("gender",axis=1)
y_data=df["gender"]
x_data=x_data.apply(pd.to_numeric)
X_train,X_test,Y_train,Y_test = train_test_split(x_data,y_data,test_size=0.2)


model=Sequential()
model.add(Dense(32,activation="relu",input_dim=7))
model.add(Dense(64,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.summary()

model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=100,validation_data=(X_test,Y_test))

with open("model.pkl",'wb') as files:
  pickle.dump(model,files)