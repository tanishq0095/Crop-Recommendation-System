from flask import Flask, request, render_template
import numpy as np
import pickle

# 1. Initialize the Flask app FIRST
app = Flask(__name__)

# 2. Load your ML models and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get data from form and convert to float
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    # Prepare feature list
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scaling - Use the scaler that matches your training
    mx_features = mx.transform(single_pred)
    
    # Predict
    prediction = model.predict(mx_features)

    # Correct Dictionary (0-indexed alphabetical)
    crop_dict = {
        0: "Apple", 1: "Banana", 2: "Blackgram", 3: "Chickpea", 4: "Coconut", 
        5: "Coffee", 6: "Cotton", 7: "Grapes", 8: "Jute", 9: "Kidneybeans", 
        10: "Lentil", 11: "Maize", 12: "Mango", 13: "Mothbeans", 14: "Mungbean", 
        15: "Muskmelon", 16: "Orange", 17: "Papaya", 18: "Pigeonpeas", 
        19: "Pomegranate", 20: "Rice", 21: "Watermelon"
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Could not determine the crop."

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)