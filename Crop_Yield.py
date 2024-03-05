import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('NewCropdataset.csv')

# Define the features and the target
features = ['Rain Fall (mm)', 'Fertilizer(urea) (kg/acre)', 'Temperature (°C)', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
target = 'Yeild (Q/acre)'
X = data[features]
y = data[target]

# Scale the features
scaler = MinMaxScaler()
X1 = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R²: {r2}')
print("\n")
print("Normalized Vales:")
print(X1)

plt.plot(X1)
print("\n")
print("Test Predictions: \n",predictions)

# Function to predict yield given the features
def predict_yield(features_input):
    # Convert the input to a DataFrame
    features_input_df = pd.DataFrame([features_input], columns=features)

    # Scale the features
    features_scaled = scaler.transform(features_input_df)

    # Predict the yield
    predicted_yield = model.predict(features_scaled)

    return predicted_yield[0]


# Test the function with an example
example_features = [1230,80,28,80,24,20]  # Replace with your actual feature values
print("\n")
print(f'Predicted Yield: {predict_yield(example_features)}')


# Predictions for the training data
train_predictions = model.predict(X_train)

# Create a figure and axis
fig, ax = plt.subplots()

# Scatter plot for training data
ax.scatter(y_train, train_predictions, edgecolors=(0, 0, 0), alpha=0.5, color='blue', label='Train')

# Scatter plot for testing data
ax.scatter(y_test, predictions, edgecolors=(0, 0, 0), alpha=0.5, color='red', label='Test')

# Set labels
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')

# Title and legend
ax.set_title('True vs Predicted Values for Training and Testing Sets')
ax.legend()

# Show the plot
plt.show()
