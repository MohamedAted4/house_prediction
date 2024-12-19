

import matplotlib
matplotlib.use('agg')
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from multiple_linear_reg import predicted_price_rf ,rf_model,scaler_X,scaler_y,predict_house_price,predict_houseprice, df


# MongoDB connection
client = MongoClient()  
# database name
db = client.DST
# table name  
Table = db.data_house_input 

data_house_input = [
    {'size': 1014, 'taxes': 3.45, 'age': 20},
    {'size': 1700, 'taxes': 9.253, 'age': 72},
    {'size': 1046, 'taxes': 4.110, 'age': 24},
    {'size': 1923, 'taxes': 8.488, 'age': 15},
    {'size': 665, 'taxes': 2.143, 'age': 19}
]

# Features and target variable
X = df[['Size', 'Taxes', 'age']]
y = df['Price']

# Scale features and target
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Prediction
predicted_prices = rf_model.predict(X_scaled)
predicted_prices_scaled = scaler_y.inverse_transform(predicted_prices.reshape(-1, 1))  # Rescale predictions

result = Table.delete_many({})
for i, house in enumerate(data_house_input):
    house['predicted_price'] = predicted_prices_scaled[i][0]  # Add predicted price to each entry
    
# Insert all data with predictions
result = Table.insert_many(data_house_input)

# return data from MongoDB
predicted_data = list(Table.find({}, {'_id': False}))  # Exclude '_id' from results

# Convert to DataFrame for display
predicted_df = pd.DataFrame(predicted_data)
print(predicted_df)