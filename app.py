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

def build_model(hp):
  model=Sequential()
  model.add(Dense(units=hp.Int("units",min_value=8,max_value=128,step=8),activation="relu",input_dim=8))
  model.add(Dense(1,activation="sigmoid"))
  model.compile(optimizer="rmsprop",loss='binary_crossentropy',metrics=["accuracy"])
  return model

mode=kt.RandomSearch(build_model,objective="val_accuracy",max_trials=5,directory="mydiv",project_name="srk")

model=Sequential()
model.add(Dense(32,activation="relu",input_dim=7))
model.add(Dense(64,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.summary()
mode.search(X_train,Y_train,epochs=5,validation_data=(X_test,Y_test))
mode.get_best_hyperparameters()[0].values
model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=100,validation_data=(X_test,Y_test))
model=tuner.get_best_models(num_models=1)[0]
with open("model.pkl",'wb') as files:
  pickle.dump(model,files)
