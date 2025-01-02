from flask import Flask, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route("/submit-data", methods=["GET"])
def submit_data():
    features = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "GenHlth","MentHlth", "PhysHlth", "DiffWalk",
        "Age", "Education","Income"
    ]
    form_data = request.args

    data = {key: form_data.get(key) for key in form_data}

    model_data=[[]]

    for feature in features:
        model_data[0].append(float(data[feature]))

    print("Received Data:", model_data[0])
    return {"status": "success", "data": data}, 200


if __name__ == "__main__":
    app.run(debug=True)
