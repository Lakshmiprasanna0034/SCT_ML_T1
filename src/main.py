# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load your dataset
df = pd.read_csv('C:/Users/Lakshmi Prasanna/SCT_ML_T1/data/house_price_regression_dataset.csv')


print(df.head())        
print(df.columns)       
#Select features and target
X = df[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms']]
y = df['House_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Making predictions on the test set
y_pred = model.predict(X_test)

#Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Printing evaluation metrics
print("Mean Squared Error:", mse)
print("R² Score:", r2)


new_house = pd.DataFrame([[2000, 3, 2]], columns=['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms'])
predicted_price = model.predict(new_house)
print("Predicted price for the new house:", predicted_price[0])

