

from flask import Flask,request
import pandas as pd
import numpy as np 
import pickle



##from wich point the application start 
app=Flask(__name__)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)


@app.route('/')
def welcome ():
	return "welcome All"

##providing those data at url
@app.route('/predict')
def predict_note_authentification():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]]) 
    
    return "The predicted value is "+ str(prediction)
    


#providing data in a csv file
@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)

    
    return "The predicted value is "+ str(list(prediction))


if __name__=='__main__':
	app.run()