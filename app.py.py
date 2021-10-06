import numpy as np
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/submit', methods=['POST'])
def submit_data():
    if request.method == 'POST':
        num1 = int(request.form['num1'])
        num2 = int(request.form['num2'])
        num3 = int(request.form['num3'])

        num4 = int(request.form['num4'])

        ar = np.array([num1, num2, num3, num4])

        pred = model.predict([ar])

    # return "Nothings"
    return render_template('home.html', result=pred)


if __name__ == '__main__':
    app.run(debug=True)
