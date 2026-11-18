from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "API ML funcionando"

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        print("JSON recibido:")
        print(data)

        df = pd.DataFrame([data])

        print(df)

        prediction = model.predict(df)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)