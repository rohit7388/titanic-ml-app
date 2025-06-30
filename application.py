from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('titanic_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    final_input = np.array([input_data])
    prediction = model.predict(final_input)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
