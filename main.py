import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

model=pickle.load(open("model.pkl", 'rb'))
app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST","GET"])
def predict():
    inps=[float(request.form.get('hair_length')),float(request.form.get('lname')),float(request.form.get('fname')),float(request.form.get('nose_wide')),float(request.form.get('nose_long')),float(request.form.get('lips_th')),float(request.form.get('dist_lip'))]
    prediction=model.predict([inps])
    for i in range(0,7):
        if((i==1) or (i==2)):
            continue
        elif(inps[i]==1):
            inps[i]="Yes"
        elif(inps[i]==0):
            inps[i]="No"
            
    param0=inps[0] 
    param1=inps[1]   
    param2=inps[2]   
    param3=inps[3] 
    param4=inps[4]   
    param5=inps[5]  
    param6=inps[6]  
    if(prediction[0]>=0.5):
        output="Male"
    else:
        output="Female"

    return render_template('index.html',prediction_text="Result predicted by model: "+output,
                param1="Forehead Width: "+str(param1),
                param0="Hair Long: "+str(param0),
                param2="Forehead Height: "+str(param2),
                param3="Nose wide: "+str(param3),
                param4="Nose long: "+str(param4),
                param5="Lips thin: "+str(param5),
                param6="Long distance between lips and nose: "+str(param6)
                )

if __name__ =="__main__":
    app.run(debug=True)