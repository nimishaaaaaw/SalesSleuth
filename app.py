from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load trained model and historical sales data
model = joblib.load('sales_predictor_model.pkl')
daily_sales = pd.read_csv('daily_sales.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_vals = {'year': '', 'month': '', 'day': '', 'weekday': ''}

    if request.method == 'POST':
        if 'datepicker' in request.form and request.form['datepicker']:
            try:
                dt = datetime.strptime(request.form['datepicker'], '%Y-%m-%d')
                year, month, day, weekday = dt.year, dt.month, dt.day, dt.weekday()
                input_vals = {'year': year, 'month': month, 'day': day, 'weekday': weekday}
            except Exception:
                input_vals = {'year': '', 'month': '', 'day': '', 'weekday': ''}
        else:
            year = int(request.form['year'])
            month = int(request.form['month'])
            day = int(request.form['day'])
            weekday = int(request.form['weekday'])
            input_vals = {'year': year, 'month': month, 'day': day, 'weekday': weekday}

        if all(str(v) for v in input_vals.values()):
            features = pd.DataFrame([[input_vals['year'], input_vals['month'], input_vals['day'], input_vals['weekday']]], 
                                    columns=['Year', 'Month', 'Day', 'Weekday'])
            prediction = round(model.predict(features)[0], 2)

    chart_data = daily_sales.tail(30)
    chart_labels = chart_data['Date'].tolist()
    chart_sales = chart_data['Sales'].tolist()

    return render_template('index.html',
                           prediction=prediction,
                           input_vals=input_vals,
                           chart_labels=chart_labels,
                           chart_sales=chart_sales)

if __name__ == '__main__':
    app.run(debug=True)
