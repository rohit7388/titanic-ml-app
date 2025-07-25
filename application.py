from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[str(i)]) for i in range(14)]
        prediction = model.predict([np.array(data)])

        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
