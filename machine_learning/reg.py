import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
url = r"C:\Users\pc\data_tools\dataset\merge\modified_file.xlsx"
df = pd.read_excel(url)



# Define features and target variable
features = ['Size', 'Taxes', 'age', 'bedroom', 'bathroom','garage']
X_train = df[features]
y_train = df['Price']


# Calculate the correlation matrix
correlation_matrix = df[features + ['Price']].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Matrix Heatmap")
plt.show()




# Add intercept for regression model
X_train_sm = sm.add_constant(X_train)

# Fit OLS model
ols_model = sm.OLS(y_train, X_train_sm).fit()

# Calculate and display ANOVA table components
y_train_mean = y_train.mean()
SST = ((y_train - y_train_mean) ** 2).sum()
SSR = ((ols_model.fittedvalues - y_train_mean) ** 2).sum()
SSE = ((y_train - ols_model.fittedvalues) ** 2).sum()
df_regression = X_train.shape[1]
df_residual = len(y_train) - df_regression - 1
MSR = SSR / df_regression
MSE = SSE / df_residual
F_stat = MSR / MSE

print("ANOVA Table:")
print(f"SST (Total Sum of Squares): {SST:.2f}")
print(f"SSR (Regression Sum of Squares): {SSR:.2f}")
print(f"SSE (Error Sum of Squares): {SSE:.2f}")
print(f"Degrees of Freedom (Regression): {df_regression}")
print(f"Degrees of Freedom (Residual): {df_residual}")
print(f"MSR (Mean Square for Regression): {MSR:.2f}")
print(f"MSE (Mean Square Error): {MSE:.2f}")
print(f"F-statistic: {F_stat:.2f}")

# Calculate R² and adjusted R²
R2 = 1 - (SSE / SST)
adjusted_R2 = 1 - ((1 - R2) * (len(y_train) - 1) / df_residual)
print("R²:", round(R2, 4))
print("Adjusted R²:", round(adjusted_R2, 4))

# Fit restricted model for Partial F-Test
restricted_features = ['Size', 'Taxes', 'age','bedroom', 'bathroom','garage']
X_train_restricted = X_train[restricted_features]
X_train_restricted_sm = sm.add_constant(X_train_restricted)
restricted_model = sm.OLS(y_train, X_train_restricted_sm).fit()

RSS_r = ((y_train - restricted_model.fittedvalues) ** 2).sum()
F_partial = ((RSS_r - SSE) / (df_regression - len(restricted_features))) / (SSE / df_residual)
print("Partial F-Test F-statistic:", round(F_partial, 4))

# Ridge regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
print("Ridge Regression Coefficients:", ridge_model.coef_)
print("Ridge Regression Intercept:", ridge_model.intercept_)

# T-Test and coefficients
print("T-Statistics for coefficients:")
coefficients = ols_model.params
standard_errors = np.sqrt(np.diag(ols_model.cov_params()))
t_values = coefficients / standard_errors
print(t_values)

print("P-Values for coefficients:")
print(ols_model.pvalues)

# Multiple Regression Formula
formula = f"Price = {coefficients[0]:.2f}"
for coef, feature in zip(coefficients[1:], X_train.columns):
    formula += f" + ({coef:.2f} * {feature})"
print("Multiple Regression Formula:")
print(formula)

# Residual plot
residuals = y_train - ols_model.fittedvalues
plt.scatter(ols_model.fittedvalues, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


##! lasso 



# Load the dataset
url = r"C:\Users\pc\data_tools\dataset\merge\modified_file.xlsx"
df = pd.read_excel(url)

# Drop unnecessary columns
df = df.drop(columns=["URL", "Type", "Levels", "Garage", "MLS Number", "address", "date released"])

# Display basic stats of the dataset
print(df.describe())

# Step 1: Prepare your features (X) and target (y)
X = df.drop('Price', axis=1)  # Features (exclude target variable 'Price')
y = df['Price']  # Target variable (Price)

# Display the first few rows and shape of the dataset
print(df.head())
print("Shape of the Dataset: {}".format(df.shape))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()

# Scale all features for both training and testing data
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test_scaled = scaler.transform(X_test)  # Transform test data based on training data

print("Shape of Train Features after Scaling: {}".format(X_train_scaled.shape))
print("Shape of Test Features after Scaling: {}".format(X_test_scaled.shape))

# GridSearchCV to find the best hyperparameter for Lasso
params = {"alpha": np.arange(0.00001, 1.0, 0.1)}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lasso = Lasso()
lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv.fit(X_train_scaled, y_train)

# Best alpha parameter
best_alpha = lasso_cv.best_params_["alpha"]
print(f"Best Alpha Parameter: {best_alpha}")

# Feature column names
names = df.drop("Price", axis=1).columns
print("Column Names: {}".format(names.values))

# Fit the Lasso model with the best alpha parameter
lasso_best = Lasso(alpha=best_alpha)
lasso_best.fit(X_train_scaled, y_train)

# Extract and plot feature importance
lasso_coefficients = np.abs(lasso_best.coef_)
plt.bar(names, lasso_coefficients)
plt.xticks(rotation=90)
plt.grid()
plt.title("Feature Selection Based on Lasso")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.ylim(0, max(lasso_coefficients) + 0.1)
plt.show()

# Select features based on importance threshold
importance_threshold = 0.001
selected_features = np.array(names)[lasso_coefficients > importance_threshold]
print(f"Selected Feature Columns: {selected_features}")

# Create a new dataset with the selected features
df_selected = df[selected_features.tolist() + ["Price"]]
print(df_selected.head())

# Get the coefficients (theta values) of the model
coefficients = lasso_best.coef_

# Print the theta values (coefficients)
print("\nTheta Values (Coefficients):")
for feature, coef in zip(selected_features, coefficients):
    print(f"{feature}: {coef}")

# Build the regression equation
equation = f"Price = {lasso_best.intercept_:.2f}"
for feature, coef in zip(selected_features, coefficients):
    equation += f" + ({coef:.2f} * {feature})"

print("\nRegression Equation from lasso :")
print(equation)

# Calculate R-squared (R²) from the Lasso model
r_squared_train = lasso_best.score(X_train_scaled, y_train)  # R² for training data
r_squared_test = lasso_best.score(X_test_scaled, y_test)    # R² for test data

print(f"R-squared (Training Data): {r_squared_train:.4f}")
print(f"R-squared (Test Data): {r_squared_test:.4f}")


##! comparison


# Ridge and Lasso Coefficients Comparison

# Fit Ridge regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_coefficients = ridge_model.coef_

# Fit Lasso regression model with the best alpha parameter
lasso_best = Lasso(alpha=best_alpha)
lasso_best.fit(X_train_scaled, y_train)
lasso_coefficients = lasso_best.coef_

# Create a DataFrame for comparison
coef_comparison = pd.DataFrame({
    'Feature': names,
    'Ridge Coefficients': ridge_coefficients,
    'Lasso Coefficients': lasso_coefficients
})

# Set up the plot
plt.figure(figsize=(14, 6))

# Plot Ridge coefficients
plt.subplot(1, 2, 1)
plt.bar(coef_comparison['Feature'], coef_comparison['Ridge Coefficients'], color='blue', alpha=0.6)
plt.xticks(rotation=90)
plt.title('Ridge Regression Coefficients')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')

# Plot Lasso coefficients
plt.subplot(1, 2, 2)
plt.bar(coef_comparison['Feature'], coef_comparison['Lasso Coefficients'], color='green', alpha=0.6)
plt.xticks(rotation=90)
plt.title('Lasso Regression Coefficients')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')

# Show the plots
plt.tight_layout()
plt.show()
