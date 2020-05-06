import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, render_template, request, redirect

# “annual_inc”, “fico_range_low”, “term”, “loan_amnt”

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
#################################################
# Flask Routes
#################################################
keys = {
    "1":"Interest rate less than 10%",
    "2":"Interest rate between 11% and 14%",
    "3":"Interest rate between 14% and 18%",
    "4":"Interest rate over 18%"
}

@app.route("/")
def welcome():
    """List all available api routes."""
    return render_template("index.html")

@app.route("/form",methods=["POST"])
def form():
    annual_inc=request.form["annual_inc"]
    fico_range_low=request.form["fico_range_low"]
    term=request.form["term"]
    loan_amnt=request.form["loan_amnt"]
    #data = [annual_inc, fico_range_low, term, loan_amnt] #request.get_json() #(force=True)
    data=pd.DataFrame({"annual_inc": [annual_inc], "fico_range_low": [fico_range_low], "term": [term], "loan_amnt": [loan_amnt]})
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)#.reshape(-1,1)

    prediction = model.predict(data_scaled) 

    output = {"prediction":keys[str(prediction[0])]}
    return render_template("index.html",output=output)

   # return jsonify(output)
   # return jsonify(data)







if __name__ == '__main__':
    app.run(debug=True)
