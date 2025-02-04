import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

plt.rcParams['font.family'] = ['Microsoft YaHei']  # Set font to support Chinese and special characters
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# Set Chinese font to SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Load the filled data set
file_path = 'C:\\Users\\沐阳\\Desktop\\Database(standardization).xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Split the data set
X = data.iloc[:, 3:]  # Features
y = data['Industrial added value']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize regression models
models = {
    'XGBoost': XGBRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Linear Regression': LinearRegression(),
    'Support Vector Machine': SVR(),
    'KNN Regression': KNeighborsRegressor()
}

# Define a function for scientific notation formatting
def scientific_format(num):
    return "{:.2e}".format(num)

# Train and evaluate regression models
results = []
for name, model in models.items():
    # Calculate R² and MSE using cross-validation
    mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=3)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=3)

    # Convert MSE scores
    mse_avg = -mse_scores.mean()
    r2_avg = r2_scores.mean()

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate R² and MSE for the test set
    r2_test = r2_score(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred)

    # Store results
    results.append({
        'Name': name,
        'CV MSE': mse_avg,
        'CV R²': r2_avg,
        'Test MSE': mse_test,
        'Test R²': r2_test
    })

    # Print results with scientific notation formatting
    print(f'{name} - CV MSE: {scientific_format(mse_avg)}, CV R²: {scientific_format(r2_avg)}, Test MSE: {scientific_format(mse_test)}, Test R²: {scientific_format(r2_test)}')

# Sort MSE values
results = sorted(results, key=lambda x: x['CV MSE'])
model_names = [result['Name'] for result in results]
sorted_mse_scores = [result['CV MSE'] for result in results]

# Visualize MSE comparison
plt.figure(figsize=(12, 8))
colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple']
plt.bar(range(len(models)), sorted_mse_scores, color=colors)
plt.xticks(range(len(models)), model_names, rotation=45)
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
#plt.title('Comparison of Regression Models')
plt.subplots_adjust(bottom=0.3)  # Adjust bottom margin
plt.show()

# Define feature and target variables
X = data.iloc[:, 3:]  # Features
y = data.iloc[:, 2]   # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate an XGBoost regressor
model = xgb.XGBRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red')
plt.fill_between(range(len(y_test)), y_pred - 1.96 * np.std(y_pred), y_pred + 1.96 * np.std(y_pred), color='gray', alpha=0.2, label='95% Confidence Interval')
plt.xlabel('Samples')
plt.ylabel('Value')
#plt.title('Actual vs Predicted with Confidence Interval')
plt.legend()
plt.show()

# Evaluate the model
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print('Root Mean Squared Error:', rmse)
print('Mean Absolute Error:', mae)
print('Mean Absolute Percentage Error:', mape)
print('R^2 Score:', r2)

# Explain model predictions
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
# Feature importance plot
shap.summary_plot(shap_values.values, X_test, plot_type='bar', max_display=25)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, max_display=25)  # Use shap_values object directly

# Call dependence_plot
feature_name = "PAT"  # Make sure this feature name is in your dataset
shap.dependence_plot(feature_name, shap_values.values, X_test)  # Use .values to get SHAP values array
shap.plots.heatmap(shap_values)
# Print the importance of all feature variables
feature_names = X_test.columns
feature_importance = np.abs(shap_values.values).mean(axis=0)
feature_importance_df = pd.DataFrame(list(zip(feature_names, feature_importance)), columns=['Feature', 'Importance'])
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)