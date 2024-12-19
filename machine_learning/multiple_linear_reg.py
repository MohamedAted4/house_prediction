import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pyfiglet
from colorama import Fore, Style

# Load the dataset
url = r"C:\Users\pc\data_tools\dataset\merge\modified_file.xlsx"
df = pd.read_excel(url)

# Function to drop outliers based on price percentiles
def remove_outliers(df, column_name, lower_percentile=0.10, upper_percentile=0.90):
    lower_bound = df[column_name].quantile(lower_percentile)
    upper_bound = df[column_name].quantile(upper_percentile)
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

# Remove outliers in the 'Price' column
df = remove_outliers(df, 'Price')

print(len(df))
print(df.describe())
# print(6.380000e+05)
# print(4.380000e+06)
##! vif
df_vif=pd.read_excel(url) 
def calculate_vif(X):
    # Add a constant to the model (intercept term)
    X_with_const = add_constant(X)
    
    # Create an empty list to store the VIFs
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    
    # Calculate the VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    
    return vif_data


X_vif = pd.DataFrame({
    'Size': df_vif['Size'],  # Features
    'Taxes': df_vif['Taxes'],
    'Age': df_vif['age'],
    'garage': df_vif['garage'],
    'bedroom': df_vif['bedroom'],
    'bathroom': df_vif['bathroom'],
})
# We only need to calculate VIF for the features, not the target (Price)
vif_result = calculate_vif(X_vif)

# Plot VIF values for features
plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='VIF', data=vif_result)
plt.xticks(rotation=45)
plt.xlabel('Feature')
plt.ylabel('VIF')
plt.title('VIF of Features')
plt.show()


##! after removing feature 

X_vif = pd.DataFrame({
    'Size': df_vif['Size'],  # Features
    'Taxes': df_vif['Taxes'],
    'Age': df_vif['age'],
    
})
# We only need to calculate VIF for the features
vif_result = calculate_vif(X_vif)

# Plot VIF values for features
plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='VIF', data=vif_result)
plt.xticks(rotation=45)
plt.xlabel('Feature')
plt.ylabel('VIF')
plt.title('VIF of Features')
plt.show()


# Drop unnecessary columns
columns_to_drop = ["URL", "Type", "Levels", "MLS Number", "address", "date released", "bedroom", "garage"]
df = df.drop(columns=columns_to_drop)

# Features and target variable
X = df[['Size', 'Taxes', 'age']]
y = df['Price']

# Scale features and target
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)) ## converts the 1D array into a 2D array of shape


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Polynomial Regression (Degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

##* Ridge Regression
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

##* Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=50)
rf_model.fit(X_train, y_train.ravel())
y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

##*Gradient Boosting Regression
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train.ravel())
y_pred_gb = gb_model.predict(X_test)

mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

##* Support Vector Regression
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train, y_train.ravel())
y_pred_svr = svr_model.predict(X_test)

mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

##*XGBoost Regression
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train.ravel())
y_pred_xgb = xgb_model.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Function to predict house price
def predict_house_price(model, scaler_X, scaler_y, size, taxes, age, is_poly=False):
    taxes = float(taxes)
    user_input = np.array([size, taxes, age]).reshape(1, -1)
    user_input_scaled = scaler_X.transform(pd.DataFrame(user_input, columns=['Size', 'Taxes', 'age']))
    if is_poly:
        user_input_scaled = poly.transform(user_input_scaled)
    prediction_scaled = model.predict(user_input_scaled)
    return scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]

# Predictions for example input
predict_houseprice = {'size': 1046 , 'taxes': 4.11, 'age': 24}
predicted_price_poly = predict_house_price(poly_model, scaler_X, scaler_y, **predict_houseprice, is_poly=True)
predicted_price_ridge = predict_house_price(ridge_model, scaler_X, scaler_y, **predict_houseprice)
predicted_price_rf = predict_house_price(rf_model, scaler_X, scaler_y, **predict_houseprice)
predicted_price_gb = predict_house_price(gb_model, scaler_X, scaler_y, **predict_houseprice)
predicted_price_svr = predict_house_price(svr_model, scaler_X, scaler_y, **predict_houseprice)
predicted_price_xgb = predict_house_price(xgb_model, scaler_X, scaler_y, **predict_houseprice)

#?? Display results


print(f"Polynomial Regression MSE: {mse_poly:.2f}, R²: {r2_poly:.2f}")
print(f"Ridge Regression MSE: {mse_ridge:.2f}, R²: {r2_ridge:.2f}")
print(f"Random Forest Regression MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")
print(f"Gradient Boosting Regression MSE: {mse_gb:.2f}, R²: {r2_gb:.2f}")
print(f"Support Vector Regression MSE: {mse_svr:.2f}, R²: {r2_svr:.2f}")
print(f"XGBoost Regression MSE: {mse_xgb:.2f}, R²: {r2_xgb:.2f}")

print(f"\nPredicted house prices for example input {predict_houseprice}:")
print(f"Polynomial Regression: ${predicted_price_poly:,.2f}")
print(f"Ridge Regression: ${predicted_price_ridge:,.2f}")
print(f"Random Forest: ${predicted_price_rf:,.2f}")
print(f"Gradient Boosting: ${predicted_price_gb:,.2f}")
print(f"Support Vector Regression: ${predicted_price_svr:,.2f}")
print(f"XGBoost: ${predicted_price_xgb:,.2f}")


##! J_train J_test 
# Compute training and testing errors for all models
def calculate_errors(model, X_train, X_test, y_train, y_test, is_poly=False):
    if is_poly:
        X_train = poly.transform(X_train)
        X_test = poly.transform(X_test)
    
    # Training predictions and error
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    
    # Testing predictions and error
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    return train_mse, test_mse

# Polynomial Regression
poly_train_mse, poly_test_mse = calculate_errors(poly_model, X_train, X_test, y_train, y_test, is_poly=True)

# Ridge Regression
ridge_train_mse, ridge_test_mse = calculate_errors(ridge_model, X_train, X_test, y_train, y_test)

# Random Forest Regression
rf_train_mse, rf_test_mse = calculate_errors(rf_model, X_train, X_test, y_train, y_test)

# Gradient Boosting Regression
gb_train_mse, gb_test_mse = calculate_errors(gb_model, X_train, X_test, y_train, y_test)

# Support Vector Regression
svr_train_mse, svr_test_mse = calculate_errors(svr_model, X_train, X_test, y_train, y_test)

# XGBoost Regression
xgb_train_mse, xgb_test_mse = calculate_errors(xgb_model, X_train, X_test, y_train, y_test)

# Print training and testing errors
print("\nTraining and Testing Errors (MSE):")
print(f"Polynomial Regression - Train MSE: {poly_train_mse:.2f}, Test MSE: {poly_test_mse:.2f}")
print(f"Ridge Regression - Train MSE: {ridge_train_mse:.2f}, Test MSE: {ridge_test_mse:.2f}")
print(f"Random Forest - Train MSE: {rf_train_mse:.2f}, Test MSE: {rf_test_mse:.2f}")
print(f"Gradient Boosting - Train MSE: {gb_train_mse:.2f}, Test MSE: {gb_test_mse:.2f}")
print(f"Support Vector Regression - Train MSE: {svr_train_mse:.2f}, Test MSE: {svr_test_mse:.2f}")
print(f"XGBoost - Train MSE: {xgb_train_mse:.2f}, Test MSE: {xgb_test_mse:.2f}")


##! Visualizations for each algorithm's performance and comparison

# Create a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot each model in the grid with MSE and R² values in the legend
axes[0, 0].scatter(y_test, y_pred_poly, color='blue', alpha=0.6, label='Polynomial Predictions')
axes[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label=f'Ideal Line\nMSE = {mse_poly:.2f}, R² = {r2_poly:.2f}')
# axes[0, 0].set_xlabel('Actual Price (scaled)')
axes[0, 0].set_ylabel('Predicted Price (scaled)')
axes[0, 0].set_title('Polynomial Regression: Predicted vs Actual')
axes[0, 0].legend()

axes[0, 1].scatter(y_test, y_pred_ridge, color='green', alpha=0.6, label='Ridge Predictions')
axes[0, 1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label=f'Ideal Line\nMSE = {mse_ridge:.2f}, R² = {r2_ridge:.2f}')
# axes[0, 1].set_xlabel('Actual Price (scaled)')
axes[0, 1].set_ylabel('Predicted Price (scaled)')
axes[0, 1].set_title('Ridge Regression: Predicted vs Actual')
axes[0, 1].legend()

axes[1, 0].scatter(y_test, y_pred_rf, color='purple', alpha=0.6, label='Random Forest Predictions')
axes[1, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label=f'Ideal Line\nMSE = {mse_rf:.2f}, R² = {r2_rf:.2f}')
# axes[1, 0].set_xlabel('Actual Price (scaled)')
axes[1, 0].set_ylabel('Predicted Price (scaled)')
axes[1, 0].set_title('Random Forest Regression: Predicted vs Actual')
axes[1, 0].legend()

axes[1, 1].scatter(y_test, y_pred_gb, color='orange', alpha=0.6, label='Gradient Boosting Predictions')
axes[1, 1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label=f'Ideal Line\nMSE = {mse_gb:.2f}, R² = {r2_gb:.2f}')
# axes[1, 1].set_xlabel('Actual Price (scaled)')
axes[1, 1].set_ylabel('Predicted Price (scaled)')
axes[1, 1].set_title('Gradient Boosting: Predicted vs Actual')
axes[1, 1].legend()

axes[2, 0].scatter(y_test, y_pred_svr, color='cyan', alpha=0.6, label='SVR Predictions')
axes[2, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label=f'Ideal Line\nMSE = {mse_svr:.2f}, R² = {r2_svr:.2f}')
# axes[2, 0].set_xlabel('Actual Price (scaled)')
axes[2, 0].set_ylabel('Predicted Price (scaled)')
axes[2, 0].set_title('Support Vector Regression: Predicted vs Actual')
axes[2, 0].legend()

axes[2, 1].scatter(y_test, y_pred_xgb, color='magenta', alpha=0.6, label='XGBoost Predictions')
axes[2, 1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label=f'Ideal Line\nMSE = {mse_xgb:.2f}, R² = {r2_xgb:.2f}')
# axes[2, 1].set_xlabel('Actual Price (scaled)')
axes[2, 1].set_ylabel('Predicted Price (scaled)')
axes[2, 1].set_title('XGBoost: Predicted vs Actual')
axes[2, 1].legend()

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the figure
plt.show()




# Final Comparative Visualization
models = ['Polynomial', 'Ridge', 'Random Forest', 'Gradient Boosting', 'SVR', 'XGBoost']
mse_scores = [mse_poly, mse_ridge, mse_rf, mse_gb, mse_svr, mse_xgb]
r2_scores = [r2_poly, r2_ridge, r2_rf, r2_gb, r2_svr, r2_xgb]

# Bar plot for MSE
plt.figure(figsize=(10, 6))
plt.bar(models, mse_scores, color=['blue', 'green', 'purple', 'orange', 'cyan', 'magenta'])
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Comparison of MSE Across Models')
plt.show()

# Bar plot for R² Scores
plt.figure(figsize=(10, 6))
plt.bar(models, r2_scores, color=['blue', 'green', 'purple', 'orange', 'cyan', 'magenta'])
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.title('Comparison of R² Scores Across Models')
plt.show()



##! addition visualization 

##! 1 Feature Importance for Random Forest
feature_importance_rf = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 6))
plt.bar(features, feature_importance_rf, color='magenta')
plt.xlabel('Importance Score')
plt.title('Feature Importance (Random Forest)')
plt.show()

##!2
sns.pairplot(df[['Size', 'Taxes', 'age', 'Price']], diag_kind='kde')
plt.suptitle("Pair Plot of Features and Target Variable", y=1.02)
plt.show()

##!3

# Ensure residuals are 1D arrays
residuals = {
    'Polynomial': (y_test - y_pred_poly.ravel()).flatten(),
    'Ridge': (y_test - y_pred_ridge.ravel()).flatten(), 
    'Random Forest': (y_test - y_pred_rf.ravel()).flatten(),
    'Gradient Boosting': (y_test - y_pred_gb.ravel()).flatten(),
    'SVR': (y_test - y_pred_svr.ravel()).flatten(),
    'XGBoost': (y_test - y_pred_xgb.ravel()).flatten()
}

# Convert to DataFrame
residuals_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in residuals.items()]))

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=residuals_df)
plt.xlabel('Models')
plt.ylabel('Residuals')
plt.title('Error Comparison Across Models')
plt.show()





from sklearn.model_selection import learning_curve

# Polynomial Regression Learning Curve
train_sizes, train_scores, test_scores = learning_curve(poly_model, X_scaled, y_scaled, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE to positive for better visualization
train_scores = -train_scores
test_scores = -test_scores

# Plotting the learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train MSE')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Test MSE')
plt.xlabel('Training Size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve for Polynomial Regression')
plt.legend()
plt.show()



print("___"*50)
###! Kmean



def decorated_print(message):
    ascii_art = pyfiglet.figlet_format(message)
    print(Style.BRIGHT + Fore.RED + ascii_art + Style.RESET_ALL)
decorated_print(" K - MEANS ")


# Selecting only the 'Price' feature for clustering
X = np.array(df['Price']).reshape(-1, 1)

# Feature scaling using MinMaxScaler
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Elbow Method to determine the optimal number of clusters
inertia = []
for n in range(5, 11):  # Testing clusters from 1 to 10
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the elbow graph
plt.plot(range(5, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()


# Selecting only the 'Price' feature for clustering
X = np.array(df['Price']).reshape(-1, 1)

# Feature scaling using MinMaxScaler
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Silhouette Score to determine the optimal number of clusters
silhouette_scores = []
for n in range(5, 11):  # Silhouette score requires at least 2 clusters
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plotting the silhouette scores
plt.plot(range(5, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Clusters')
plt.show()



##! kmean ABCDE



X = np.array(df['Price']).reshape(-1, 1)

# Feature scaling using MinMaxScaler
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Apply KMeans clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Add cluster centers back to the original scale
cluster_centers = scaler_X.inverse_transform(kmeans.cluster_centers_).flatten()

# Sort clusters by ascending order of cluster center prices
sorted_clusters = np.argsort(cluster_centers)

# Map clusters to class labels in ascending order
cluster_labels = {sorted_clusters[i]: chr(65 + i) for i in range(len(sorted_clusters))}
df['Class'] = df['Cluster'].map(cluster_labels)

# Sort cluster_centers for plotting
sorted_cluster_centers = cluster_centers[sorted_clusters]

# Plot the data points and cluster centers
plt.figure(figsize=(12, 6))

for cluster in range(5):
    cluster_data = df[df['Cluster'] == sorted_clusters[cluster]]
    plt.scatter(
        cluster_data['Price'],
        [cluster_labels[sorted_clusters[cluster]]] * len(cluster_data),
        label=f'Class {cluster_labels[sorted_clusters[cluster]]}',
        alpha=0.7
    )

# Plot sorted cluster centers with labels
for i, center in enumerate(sorted_cluster_centers):
    plt.scatter(center, chr(65 + i), color='red', s=200, marker='x')
    plt.text(center + 0.5, chr(65 + i), f'Center {chr(65 + i)}', fontsize=10, color='black')

# Adding labels and improvements
plt.xlabel('Price')
plt.ylabel('Class')
plt.title('KMeans Clustering of Houses by Price (Ascending Order)')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()





##! scatter the point 

# Apply KMeans clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Add cluster centers back to the original scale
cluster_centers = scaler_X.inverse_transform(kmeans.cluster_centers_)

# Sort clusters by ascending order of cluster center values
sorted_clusters = np.argsort(cluster_centers.flatten())

# Map clusters to class labels (A-E) based on sorted order
cluster_labels = {sorted_clusters[i]: chr(65 + i) for i in range(len(sorted_clusters))}
df['Class'] = df['Cluster'].map(cluster_labels)

# Sort cluster_centers for correct display order
sorted_cluster_centers = cluster_centers[sorted_clusters]

# Take price
try:
    y_value = int(predicted_price_rf)  
except ValueError:
    exit()

# Scale the user input
y_scaled = scaler_X.transform([[y_value]])

# Predict the cluster for the input price
predicted_cluster = kmeans.predict(y_scaled)[0]
predicted_class = cluster_labels[predicted_cluster]

print(f"The house price {y_value} belongs to Class {predicted_class}.")

# Plot the data points and cluster centers
plt.figure(figsize=(12, 6))

for cluster in range(5):
    cluster_data = df[df['Cluster'] == sorted_clusters[cluster]]
    plt.scatter(
        cluster_data['Price'],
        [cluster_labels[sorted_clusters[cluster]]] * len(cluster_data),
        label=f'Class {cluster_labels[sorted_clusters[cluster]]}',
        alpha=0.7
    )

# Plot cluster centers with labels
for i, center in enumerate(sorted_cluster_centers):
    plt.scatter(center, cluster_labels[sorted_clusters[i]], color='red', s=200, marker='x')
    plt.text(center + 0.5, cluster_labels[sorted_clusters[i]], f'Center {cluster_labels[sorted_clusters[i]]}', fontsize=10, color='black')

# Highlight the input price as a gold point for our predictive point
plt.scatter(y_value, predicted_class, color='gold', s=150, label=f'house Price ({y_value})', marker='o')

# Adding labels and improvements
plt.xlabel('Price')
plt.ylabel('Class')
plt.title('KMeans Clustering of Houses by Price (Ascending Order)')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


##! display number for house belong that belong to each cluster
# Plot cluster distribution
plt.figure(figsize=(8, 6))
df['Class'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.xlabel('Cluster/Class')
plt.ylabel('Number of Houses')
plt.title('Distribution of Houses Across Clusters')
plt.xticks(rotation=0)
plt.show()

print("___"*50)

##! mongo DB

def decorated_print(message):
    ascii_art = pyfiglet.figlet_format(message)
    print(Style.BRIGHT + Fore.RED + ascii_art + Style.RESET_ALL)
decorated_print(" MONGO - DB ")



import matplotlib
matplotlib.use('agg')
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# MongoDB connection
client = MongoClient()
db = client.DST
Table = db.data_house_input

# Sample training data
df = pd.DataFrame({
    'Size': [1014, 1700, 1046, 1923, 665],
    'Taxes': [3.45, 9.253, 4.110, 8.488, 2.143],
    'age': [20, 72, 24, 15, 19],
    'Price': [1200000, 2500000, 1267139, 3000000, 800000]
})

# Features and target variable
X = df[['Size', 'Taxes', 'age']]
y = df['Price']

# Scale features and target
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=50)
rf_model.fit(X_scaled, y_scaled.ravel())

# Prepare input data for prediction
X_input = scaler_X.transform(df[['Size', 'Taxes', 'age']])

# Predict and rescale prices
predicted_scaled = rf_model.predict(X_input)
predicted_prices = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1))

# Add predictions to input data
for i in range(len(df)):
    df.at[i, 'predicted_price'] = round(predicted_prices[i][0], 2)

# Insert into MongoDB
Table.delete_many({})
Table.insert_many(df.to_dict('records'))

# Fetch and display predicted data
predicted_data = list(Table.find({}, {'_id': False}))
predicted_df = pd.DataFrame(predicted_data)
print(predicted_df)