from flask import Flask, request
from flask_cors import CORS
import pandas as pd
import joblib
app = Flask(__name__)
CORS(app)

scaler = joblib.load('../scaler/scaler.pkl')
model = joblib.load('../models/adaboost_model.pkl')


def transform_BMI(bmi):
    if bmi <= 18:
        return 0
    elif bmi < 25:
        return 1 / 3
    elif bmi < 30:
        return 2 / 3
    else:
        return 1

def extract_features(data):
    data["BMI"] = data["BMI"].apply(transform_BMI)
    data["health_score"] = data["HighBP"] + data["HighChol"] + data["BMI"]
    data['MultiRisk_factors'] = (data["HighBP"].astype(bool) & data["HighChol"].astype(bool) & (data["BMI"] == 1.0)).astype(int)
    data['EatingHealthyFood'] = data["Veggies"] + data["Fruits"]
    data['SocioeconomicStatus'] = data['Income'] + data['Education']
    return data

def scale_data(data:pd.DataFrame):
    f = ["PhysHlth", "MentHlth", "Age", "GenHlth"]
    print(data.columns)
    data[f] = scaler.transform(data[f])
    return data

def delete_features(data):
    features = ['Veggies', 'Fruits']
    data=data.drop(columns=features)
    return data
@app.route("/submit-data", methods=["GET"])
def submit_data():
    form_data = request.args
    try:
        data = {key: form_data.get(key) for key in form_data}
    except ValueError:
        return {"status": "error", "message": "Invalid input data. Ensure all values are numeric."}, 400

    data_df = pd.DataFrame([data]).astype(float)

    try:
        data_df = extract_features(data_df)
        data_df = scale_data(data_df)
        data_df = delete_features(data_df)

        prediction = model.predict(data_df).tolist()
        prediction_proba = model.predict_proba(data_df).tolist()
    except Exception as e:
        return {"status": "error", "message": f"Processing or prediction error: {str(e)}"}, 500

    return {"status": "success", "data": {"prediction": prediction, "probabilities": prediction_proba}}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)