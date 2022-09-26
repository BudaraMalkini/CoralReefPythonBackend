from flask import Flask, render_template, request
import pickle
import numpy as np

# read pkl files
flask_app = Flask(__name__)
model = pickle.load(open("CoralReef.pkl", "rb"))
model1 = pickle.load(open("Fisheries.pkl", "rb"))


# render prediction page.
@flask_app.route("/")
def index():
    return render_template("Prediction.html")


# Create API to pass data for front end
@flask_app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    prediction2 = model1.predict(features)
    # Give output without []
    return render_template("Prediction.html",
                           prediction_text="According to the user selected details Coral Reef Prediction is in Square kilometre (Km2): {}".format(
                               round(prediction[0], 2)),
                           prediction_text1="According to the user selected details Fisheries Prediction is in Metric Ton(MT):  {}".format(
                               round(prediction2[0], 2)))

    # return render_template("prediction.html", prediction_text = "Mangrove prediction in Ha  {}".format(prediction), prediction_text1 = "Fisheries Prediction in MT  {}".format(prediction2))


if __name__ == "__main__":
    flask_app.run(debug=True)
