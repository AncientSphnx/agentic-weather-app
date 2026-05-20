import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. Load and preprocess the dataset
df = pd.read_csv('DailyDelhiClimateTest.csv')

# Convert date to day-of-year and extract more features
df['date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month

# Add cyclical features for better seasonal patterns
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Fix outlier in meanpressure
df.loc[df['meanpressure'] < 900, 'meanpressure'] = df['meanpressure'].mean()

# Add interaction features
df['humidity_pressure'] = df['humidity'] * df['meanpressure'] / 1000

# Add lag features (previous temperatures)
df['temp_lag_1'] = df['meantemp'].shift(1).fillna(df['meantemp'].mean())
df['temp_lag_3'] = df['meantemp'].shift(3).fillna(df['meantemp'].mean())

# 2. Enhanced Features & Label
feature_columns = ['day_of_year', 'humidity', 'wind_speed', 'meanpressure',
                   'day_sin', 'day_cos', 'month_sin', 'month_cos',
                   'humidity_pressure', 'temp_lag_1', 'temp_lag_3']

X = df[feature_columns]
y = df['meantemp']

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Comparison and Selection
print("🤖 Training multiple models for comparison...")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

best_model = None
best_score = -float('inf')
best_model_name = ""

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    print(f"\n{name}:")
    print(f"   MSE: {mse:.3f}")
    print(f"   MAE: {mae:.3f}")
    print(f"   R²: {r2:.3f}")
    print(f"   CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Track best model
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

print(f"\n🏆 Best model: {best_model_name} with R² = {best_score:.3f}")

# 6. Hyperparameter Tuning for Best Model
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print(f"\n⚡ Hyperparameter tuning for {best_model_name}...")
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        model = RandomForestRegressor(random_state=42)
    else:  # Gradient Boosting
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        model = GradientBoostingRegressor(random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Final evaluation
    y_pred_tuned = best_model.predict(X_test)
    tuned_r2 = r2_score(y_test, y_pred_tuned)
    print(f"Tuned R²: {tuned_r2:.3f}")

# 7. Final Predictions and Evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Final Model Performance:")
print(f"   Mean Squared Error: {mse:.3f}")
print(f"   Mean Absolute Error: {mae:.3f}")
print(f"   R² Score: {r2:.3f}")

# 8. Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n📊 Feature Importance:")
    for i in range(len(feature_columns)):
        print(f"   {i+1}. {feature_columns[indices[i]]}: {importances[indices[i]]:.3f}")

# 9. Show some predictions
print(f"\n🔍 Sample Predictions:")
for i in range(min(5, len(X_test))):
    actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
    predicted = y_pred[i]
    print(f"   Actual: {actual:.2f}°C, Predicted: {predicted:.2f}°C, Error: {abs(actual - predicted):.2f}°C")

# 10. Custom Input with validation
print(f"\n🌡️ Predict temperature based on your own input:")

def get_user_input():
    try:
        day_of_year = float(input("Enter day of year (1-365): "))
        humidity = float(input("Enter humidity (%): "))
        wind_speed = float(input("Enter wind speed (km/h): "))
        pressure = float(input("Enter pressure (hPa): "))
        
        # Validate inputs
        if not (1 <= day_of_year <= 365):
            raise ValueError("Day of year must be between 1 and 365")
        if not (0 <= humidity <= 100):
            raise ValueError("Humidity must be between 0 and 100")
        if not (0 <= wind_speed <= 50):
            raise ValueError("Wind speed must be between 0 and 50 km/h")
        if not (900 <= pressure <= 1050):
            raise ValueError("Pressure must be between 900 and 1050 hPa")
        
        return day_of_year, humidity, wind_speed, pressure
    except ValueError as e:
        print(f"❌ Error: {e}")
        return None

user_input = get_user_input()

if user_input:
    day_of_year, humidity, wind_speed, pressure = user_input
    
    # Create additional features for prediction
    month = int((day_of_year - 1) / 30.44) % 12 + 1
    day_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_cos = np.cos(2 * np.pi * day_of_year / 365)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    humidity_pressure = humidity * pressure / 1000
    
    # Use average lag values for prediction (since we don't have historical data)
    temp_lag_1 = df['temp_lag_1'].mean()
    temp_lag_3 = df['temp_lag_3'].mean()
    
    # Create input array with all features
    custom_input = np.array([[day_of_year, humidity, wind_speed, pressure,
                             day_sin, day_cos, month_sin, month_cos,
                             humidity_pressure, temp_lag_1, temp_lag_3]])
    
    # Scale the input
    custom_input_scaled = scaler.transform(custom_input)
    
    # Make prediction
    predicted_temp = best_model.predict(custom_input_scaled)[0]
    
    print(f"\n🌡️ Predicted Temperature: {predicted_temp:.2f} °C")
    
    # Add confidence interval based on model performance
    confidence_interval = 1.96 * np.sqrt(mae)  # 95% confidence interval
    print(f"📏 95% Confidence Interval: ±{confidence_interval:.2f} °C")
    print(f"🌡️ Temperature Range: {predicted_temp - confidence_interval:.2f}°C to {predicted_temp + confidence_interval:.2f}°C")

# 11. Visualization
plt.figure(figsize=(15, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title('Actual vs Predicted')
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Temperature (°C)')
plt.ylabel('Residuals (°C)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Plot 3: Temperature over time
plt.subplot(1, 3, 3)
plt.plot(df['day_of_year'][:len(y_test)], y_test, 'b-', label='Actual', alpha=0.7)
plt.plot(df['day_of_year'][:len(y_test)], y_pred, 'r--', label='Predicted', alpha=0.7)
plt.xlabel('Day of Year')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Trends')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🎉 Model training complete! Using {best_model_name} for predictions.")