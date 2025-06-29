#HTML Code
%%writefile liver_cirrhosis_prediction_app/templates/index.html
<!DOCTYPE html>
<html>
<head>
    <title>Liver Cirrhosis Prediction</title>
</head>
<body>
    <h1>Liver Cirrhosis Prediction</h1>
    <form method="POST" action="/predict">
        <label for="feature1">Feature 1:</label><br>
        <input type="text" id="feature1" name="feature1"><br><br>
        <label for="feature2">Feature 2:</label><br>
        <input type="text" id="feature2" name="feature2"><br><br>
        <label for="feature3">Feature 3:</label><br>
        <input type="text" id="feature3" name="feature3"><br><br>
        <label for="feature4">Feature 4:</label><br>
        <input type="text" id="feature4" name="feature4"><br><br>
        <input type="submit" value="Predict">
    </form>
</body>
</html>

%%writefile liver_cirrhosis_prediction_app/templates/result.html
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h1>Prediction Result</h1>
    <p>The predicted result is: {{ prediction }}</p>
</body>
</html>

#app.py file
%%writefile liver_cirrhosis_prediction_app/app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
try:
    with open('gaussiannb_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Error: gaussiannb_model.pkl not found. The model will not be available.")
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Cannot make predictions.", 500
    try:
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])
        features = np.array([[feature1, feature2, feature3, feature4]])
        prediction = model.predict(features)[0]
        prediction_label = "Class 0" if prediction == 0 else ("Class 1" if prediction == 1 else "Class 2")
    except Exception as e:
        return f"Error processing input or making prediction: {e}", 400
    return render_template('result.html', prediction=prediction_label)
if __name__ == '__main__':
    app.run(debug=True, port=5000)
