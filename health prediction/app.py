from flask import Flask, request, jsonify, session, render_template
import pandas as pd
import numpy as np
import pickle
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')

# Load the trained model
MODEL_PATH = 'model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# Health suggestions mapping
suggestions_map = {
    'Flu': ['Drink warm fluids', 'Take rest', 'Consult doctor if symptoms worsen', 'Take over-the-counter flu medicine'],
    'Cold': ['Stay hydrated', 'Take Vitamin C', 'Gargle with salt water', 'Use nasal spray'],
    'Migraine': ['Rest in a dark, quiet room', 'Apply cold compress to forehead', 'Stay hydrated', 'Avoid bright lights'],
    'Food Poisoning': ['Drink plenty of fluids', 'Eat bland foods', 'Avoid dairy and caffeine', 'Consult doctor if vomiting persists'],
    'Pneumonia': ['Consult a doctor immediately', 'Get plenty of rest', 'Use a humidifier', 'Take prescribed medications'],
    'Fatigue': ['Maintain a regular sleep schedule', "Stay active but don't overdo it", 'Eat balanced meals', 'Reduce stress'],
    'Healthy': ['Maintain your healthy lifestyle!', 'Continue balanced diet', 'Exercise regularly', 'Stay hydrated'],
    'Severe Infection': ['URGENT: Consult a doctor immediately', 'Monitor temperature', 'Rest completely', 'Seek hospital care if needed'],
    'Common Cold': ['Rest well', 'Hydrate often', 'Warm drinks', 'Humidifier use'],
    'Headache': ['Rest', 'Stay hydrated', 'Avoid screens', 'Stress management']
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error="Model not loaded. Please train the model first.")
    
    try:
        # Get symptoms from form
        symptoms = [
            int(request.form.get('fever', 0)),
            int(request.form.get('headache', 0)),
            int(request.form.get('cough', 0)),
            int(request.form.get('fatigue', 0)),
            int(request.form.get('vomiting', 0)),
            int(request.form.get('cold', 0))
        ]
        
        # Predict
        prediction = model.predict([symptoms])[0]
        
        # Get confidence (probability)
        probabilities = model.predict_proba([symptoms])[0]
        confidence = round(np.max(probabilities) * 100, 2)
        
        # Get suggestions
        suggestions = suggestions_map.get(prediction, ["Consult a healthcare professional for advice."])
        
        # Store in session history
        if 'history' not in session:
            session['history'] = []
        
        session['history'].append({'disease': prediction, 'confidence': confidence})
        session.modified = True
        
        return render_template('index.html', 
                               prediction=prediction, 
                               confidence=confidence, 
                               suggestions=suggestions)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

@app.route('/api', methods=['GET', 'POST'])
def api():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if request.method == 'POST':
        data = request.json
        symptoms = [
            data.get('fever', 0),
            data.get('headache', 0),
            data.get('cough', 0),
            data.get('fatigue', 0),
            data.get('vomiting', 0),
            data.get('cold', 0)
        ]
    else:
        # For GET /api, return dummy or help info
        return jsonify({
            'message': 'AI Health Prediction API',
            'usage': 'POST to this endpoint with symptoms JSON',
            'example': {'fever': 1, 'headache': 0, 'cough': 1, 'fatigue': 1, 'vomiting': 0, 'cold': 0}
        })

    try:
        prediction = model.predict([symptoms])[0]
        probabilities = model.predict_proba([symptoms])[0]
        confidence = round(np.max(probabilities) * 100, 2)
        suggestions = suggestions_map.get(prediction, ["Consult a healthcare professional."])
        
        return jsonify({
            'predicted_disease': prediction,
            'confidence': f"{confidence}%",
            'suggestions': suggestions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8888)
