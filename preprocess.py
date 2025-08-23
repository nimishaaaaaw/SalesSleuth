import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
data = pd.read_csv('data.csv', encoding='ISO-8859-1')

# Convert InvoiceDate to datetime (auto-detect format)
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], dayfirst=False)

# 1. Remove rows with negative or zero quantity or unit price (likely returns/refunds)
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]

# 2. Drop rows with missing Description or CustomerID
data = data.dropna(subset=['Description', 'CustomerID'])

# 3. Calculate Sales after cleaning
data['Sales'] = data['Quantity'] * data['UnitPrice']

# 4. Extract date features
data['Year'] = data['InvoiceDate'].dt.year
data['Month'] = data['InvoiceDate'].dt.month
data['Day'] = data['InvoiceDate'].dt.day
data['Weekday'] = data['InvoiceDate'].dt.weekday  # Monday=0
data['Hour'] = data['InvoiceDate'].dt.hour
data['Quarter'] = data['InvoiceDate'].dt.quarter

# 5. One-hot encode Country
data = pd.get_dummies(data, columns=['Country'], drop_first=True)

# 6. Label encode Description due to many unique values
le = LabelEncoder()
data['Description_enc'] = le.fit_transform(data['Description'])

# 7. Scale numerical features (Quantity, UnitPrice, Sales)
scaler = StandardScaler()
data[['Quantity', 'UnitPrice', 'Sales']] = scaler.fit_transform(data[['Quantity', 'UnitPrice', 'Sales']])

# 8. Aggregate daily total sales using filtered data (only positive sales)
daily_sales = data.groupby(data['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
daily_sales.rename(columns={'InvoiceDate': 'Date'}, inplace=True)

# Display info for verification
print("\nPreprocessed data info:")
print(data.info())

print("\nSample preprocessed data:")
print(data.head())

print("\nSample aggregated daily sales:")
print(daily_sales.head())

# Save preprocessed data and daily sales for modeling
data.to_csv('preprocessed_sales_data.csv', index=False)
daily_sales.to_csv('daily_sales.csv', index=False)
