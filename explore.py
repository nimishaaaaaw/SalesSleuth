import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data (adjust path if needed)
data = pd.read_csv('data.csv', encoding='ISO-8859-1')

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Display shape and columns
print(f"\nDataset shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")

# Display data types
print("\nData types:")
print(data.dtypes)

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Convert InvoiceDate to datetime

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], dayfirst=False)


# Create 'Sales' column = Quantity * UnitPrice
data['Sales'] = data['Quantity'] * data['UnitPrice']

# Aggregate daily sales
daily_sales = data.groupby(data['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
daily_sales.rename(columns={'InvoiceDate': 'Date'}, inplace=True)

# Plot total sales over time
plt.figure(figsize=(14,6))
sns.lineplot(x='Date', y='Sales', data=daily_sales)
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top 10 countries by total sales
country_sales = data.groupby('Country')['Sales'].sum().sort_values(ascending=False).head(10)

print("\nTop 10 countries by total sales:")
print(country_sales)

# Plot sales distribution by country (top 10)
top_countries = country_sales.index.tolist()
data_top_countries = data[data['Country'].isin(top_countries)]

plt.figure(figsize=(12,6))
sns.boxplot(x='Country', y='Sales', data=data_top_countries)
plt.title('Sales Distribution in Top 10 Countries')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top 10 products by total sales
top_products = data.groupby('Description')['Sales'].sum().sort_values(ascending=False).head(10)

print("\nTop 10 products by sales:")
print(top_products)
