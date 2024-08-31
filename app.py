from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('D:\Machine learning\house_price_prediction\model\house_price_model.pkl')
scaler = joblib.load('D:\Machine learning\house_price_prediction\model\scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return render_template('index.html', prediction_text=f'Estimated House Price: ${prediction[0]:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
