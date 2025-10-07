from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load the trained ML model
best_model = joblib.load('RandomForest.pkl')

# Create Flask app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    feature_names = ['r', 'i', 'z', 'petroRad_g', 'petroRad_r', 
                     'petroR50_u', 'petroR50_g', 'petroR50_i', 
                     'petroR50_r', 'petroR50_z']

    # Get values from form and convert to float
    input_values = []
    for feature in feature_names:
        value = float(request.form[feature])  # convert string to float
        input_values.append(value)
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_values], columns=feature_names)
    
    # Predict
    prediction = best_model.predict(input_df)[0]
    
    # Render result page
    return render_template('inner-page.html', prediction=prediction)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
