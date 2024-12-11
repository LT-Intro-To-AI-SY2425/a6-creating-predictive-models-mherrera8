import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("part2-training-testing-data/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Create your training and testing datasets:
model = LinearRegression().fit(x,y)
# Use reshape to turn the x values into 2D arrays:
x.reshape(-1,1)
# Create the model
model = LinearRegression().fit(x,y)

# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_[0]), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)

# Print out the linear equation and r squared value:
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")

'''
**********TEST THE MODEL**********
'''
# reshape the xtest data into a 2D array
xtrain = x

# get the predicted y values for the xtest values - returns an array of the results

xtrain.push(model.predict(y))
# round the value in the np array to 2 decimal places
xtrain = [[round (n,2) for n in row]for row in xtrain]

# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")
plt.figure(figsize = (6,4))
plt.scatter(x,y, c="purple")

plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")
plt.legend()
plt.show()
'''
**********CREATE A VISUAL OF THE RESULTS**********
'''