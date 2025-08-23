import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy import sqrt
import joblib

# Load daily sales data
data = pd.read_csv('daily_sales.csv')

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Create date features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Weekday'] = data['Date'].dt.weekday

# Features and target
X = data[['Year', 'Month', 'Day', 'Weekday']]
y = data['Sales']

# Split train-test (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

# Train Linear Regression
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Train Random Forest
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate models
def print_metrics(y_true, y_pred, model_name):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

print_metrics(y_test, y_pred_lr, 'Linear Regression')
print_metrics(y_test, y_pred_rf, 'Random Forest')

# Save best model (example: Random Forest)
joblib.dump(rf, 'sales_predictor_model.pkl')

print("Model saved as sales_predictor_model.pkl")
