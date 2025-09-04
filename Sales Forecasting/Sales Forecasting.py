# Task 7: Sales Forecasting 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Dataset Loading & Preparation
df = pd.read_csv(r"C:\Users\AR FAST\Documents\python mini projects\Elevvo Ml tasks\train.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 2. Feature Engineering
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Lag features
df['weekly_sales_lag1'] = df['Weekly_Sales'].shift(1)
df['weekly_sales_lag2'] = df['Weekly_Sales'].shift(2)
df = df.dropna()  

# Encode categorical
le = LabelEncoder()
df['IsHoliday'] = le.fit_transform(df['IsHoliday'])

# 3. Train-Test Split (time-order preserved)
train = df[df['Date'].dt.year < 2012]
test = df[df['Date'].dt.year == 2012]

feature_cols = ['Store', 'Dept', 'Day', 'Month', 'Year', 'IsHoliday', 'weekly_sales_lag1', 'weekly_sales_lag2']
X_train = train[feature_cols]
y_train = train['Weekly_Sales']
X_test = test[feature_cols]
y_test = test['Weekly_Sales']

# 4. Scale numeric features
numeric_cols = feature_cols  
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 5. Modeling
# Linear Regression
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)
weekly_sales_linear = linear_reg_model.predict(X_test)

# Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
regressor.fit(X_train, y_train)
weekly_sales_reg = regressor.predict(X_test)

# 6. Evaluation
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return [model_name, mae, mse, rmse, r2]

results = pd.DataFrame([
    evaluate_model(y_test, weekly_sales_linear, "Linear Regression"),
    evaluate_model(y_test, weekly_sales_reg, "Random Forest Regressor")
], columns=["Model", "Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "R2 Score"])

# 7. Visualization: Model Comparison
import matplotlib.pyplot as plt

metrics = ["Mean Absolute Error","Mean Squared Error","Root Mean Squared Error","R2 Score"]
colors = ['skyblue','salmon']  

fig, axs = plt.subplots(2, 2, figsize=(13,11))  

axs = axs.flatten()  

for i, metric in enumerate(metrics):
    results.plot(x="Model", y=metric, kind="bar", ax=axs[i], rot=0, color=colors, legend=False)
    axs[i].set_title(f"{metric} Comparison", fontsize=14)
    axs[i].set_ylabel(metric, fontsize=12)
    axs[i].set_xlabel('')  
    axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # remove model labels
   
    for j, val in enumerate(results[metric]):
        axs[i].text(j, val*1.01, round(val,2), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()



# 8. Visualization: Monthly Average Sales vs Predictions
df_test = test[['Date','Weekly_Sales']].copy()
df_test['Month'] = df_test['Date'].dt.to_period('M')
monthly_actual = df_test.groupby('Month')['Weekly_Sales'].mean()

monthly_pred_linear = pd.Series(weekly_sales_linear, index=test['Date']).resample('M').mean()
monthly_pred_rf = pd.Series(weekly_sales_reg, index=test['Date']).resample('M').mean()

plt.figure(figsize=(12,6))
plt.plot(monthly_actual.index.astype(str), monthly_actual, label='Actual', color='blue')
plt.plot(monthly_pred_linear.index.astype(str), monthly_pred_linear, label='Linear Regression', color='red')
plt.plot(monthly_pred_rf.index.astype(str), monthly_pred_rf, label='Random Forest', color='green')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Average Weekly Sales')
plt.title('Monthly Average Sales: Actual vs Predicted')
plt.legend()
plt.show()
